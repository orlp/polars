mod emit;
mod pre_agg;
mod sink;
mod table;
mod tuning;

use std::sync::Arc;

use futures::FutureExt;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_core::frame::DataFrame;
use polars_core::prelude::PlRandomState;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::groups::Grouper;
use polars_expr::reduce::GroupedReduction;
use polars_ooc::MostRecentSpillContext;
use polars_utils::hashing::HashPartitioner;
use polars_utils::pl_str::PlSmallStr;
use tokio::sync::mpsc::channel;

use self::emit::Emit;
use self::pre_agg::SplitPreAggs;
use self::sink::GroupBySink;
use self::table::{GroupByParams, PartitionTable};
use self::tuning::GroupByTuning;
use super::ComputeNode;
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::graph::PortState;
use crate::hash_shuffle::keyed_frame::KeyedSpillFrame;
use crate::hash_shuffle::rounds::{
    LoadedFrame, PrepareFrame, build_in_rounds, for_each_parallel, make_waves,
};
use crate::hash_shuffle::store::StoredPartition;
use crate::metrics::{Metric, MetricUnit, NodeMetricsRegistry, kind};
use crate::nodes::group_by::InputPayload;
use crate::pipe::{RecvPort, SendPort};

enum GroupByState {
    Sink(GroupBySink),
    Emit(Emit),
    Done,
}

pub struct GroupByNode {
    state: GroupByState,
    params: Arc<GroupByParams>,
    num_inputs: usize,
    num_pipelines: usize,
    tuning: GroupByTuning,
    spill_ctx: MostRecentSpillContext,
    estimated_groups: Metric<kind::Sum>,
    actual_groups: Metric<kind::Sum>,
}

impl GroupByNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key_schema: Arc<Schema>,
        // Input stream i selects keys with key_selectors_per_input[i].
        key_selectors_per_input: Vec<Vec<StreamExpr>>,
        // Input stream i feeds grouped_reductions[k] for each k in reductions_per_input[i].
        reductions_per_input: Vec<Vec<usize>>,
        grouper: Box<dyn Grouper>,
        // grouped_reductions[k] is passed input cols grouped_reduction_cols[k].
        grouped_reduction_cols: Vec<Vec<PlSmallStr>>,
        payload_per_input: Vec<InputPayload>,
        grouped_reductions: Vec<Box<dyn GroupedReduction>>,
        output_schema: Arc<Schema>,
        random_state: PlRandomState,
        num_pipelines: usize,
        has_order_sensitive_agg: bool,
        metrics_registry: NodeMetricsRegistry,
    ) -> Self {
        let config = polars_config::config();
        let hot_table_size = (config.hot_table_size() as usize)
            .next_power_of_two()
            .max(2);
        let max_hot_table_size = (config.max_hot_table_size() as usize)
            .next_power_of_two()
            .max(hot_table_size);
        let num_inputs = key_selectors_per_input.len();
        let sink = GroupBySink::new(
            key_selectors_per_input,
            random_state,
            hot_table_size,
            max_hot_table_size,
            &grouped_reductions,
            num_pipelines,
        );
        let params = GroupByParams {
            key_schema,
            output_schema,
            reductions_per_input,
            grouped_reduction_cols,
            payload_per_input,
            grouper,
            grouped_reductions,
            has_order_sensitive_agg,
        };
        Self {
            state: GroupByState::Sink(sink),
            params: Arc::new(params),
            num_inputs,
            num_pipelines,
            tuning: GroupByTuning::from_env(),
            spill_ctx: MostRecentSpillContext::new(
                "group-by".into(),
                metrics_registry.task_metrics(),
            ),
            estimated_groups: metrics_registry
                .new_counter("group_by.estimated_groups", MetricUnit::Unit),
            actual_groups: metrics_registry.new_counter("group_by.actual_groups", MetricUnit::Unit),
        }
    }

    /// Builds the results once all input is received. With at most one partition per pipeline
    /// they are built here; otherwise the partitions are written out and built while the
    /// results are sent.
    fn finish(&self, sink: GroupBySink, state: &StreamingExecutionState) -> PolarsResult<Emit> {
        let (stats, frames, pre_aggs) = sink.finish_input(&self.params, &self.spill_ctx, state)?;
        let num_partitions = self.tuning.num_partitions(&stats, self.num_pipelines);
        let partitioner = HashPartitioner::new(num_partitions, 0);
        let pre_aggs = SplitPreAggs::new(pre_aggs, &partitioner, state)?;
        let num_frames = frames.len();
        let waves = make_waves(frames, self.tuning.wave_bytes());
        let reserve = (stats.estimated_groups * 5 / 4 / num_partitions as u64) as usize;
        let direct = num_partitions <= self.num_pipelines;
        if polars_config::config().verbose() {
            let path = if direct { "direct" } else { "written-out" };
            eprintln!(
                "[group-by]: {path} path, {num_partitions} partitions, {num_frames} frames in {} waves, {} pre-aggregates, {} estimated groups",
                waves.len(),
                pre_aggs.len(),
                stats.estimated_groups
            );
        }
        self.estimated_groups
            .reporter()
            .add(stats.estimated_groups as i64);
        let actual_groups = self.actual_groups.reporter();

        let (outputs, stored, pre_aggs) = if direct {
            let outputs = self.build_direct(pre_aggs, waves, &partitioner, reserve, state)?;
            actual_groups.add(outputs.iter().map(|out| out.height() as i64).sum());
            (outputs, Vec::new(), None)
        } else {
            let stored = self.write_out(waves, &partitioner, state)?;
            (Vec::new(), stored, Some(pre_aggs))
        };
        Ok(Emit::new(
            outputs,
            stored,
            pre_aggs,
            self.num_pipelines,
            self.params.clone(),
            reserve,
            actual_groups,
        ))
    }

    /// Builds every partition: first from the pre-aggregates, then from the cold frames one wave
    /// at a time. Returns the non-empty results, or one empty result if all are empty.
    fn build_direct(
        &self,
        pre_aggs: SplitPreAggs,
        waves: Vec<Vec<(usize, KeyedSpillFrame)>>,
        partitioner: &HashPartitioner,
        reserve: usize,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Vec<DataFrame>> {
        let num_partitions = partitioner.num_partitions();
        let mut partitions: Vec<(usize, PartitionTable, DataFrame)> = (0..num_partitions)
            .map(|p| {
                let table = PartitionTable::new(self.params.clone(), reserve);
                (p, table, DataFrame::empty())
            })
            .collect();
        for_each_parallel(
            &mut partitions,
            &|(p, table, _): &mut (usize, PartitionTable, DataFrame)| -> PolarsResult<()> {
                for (pre_agg, groups) in pre_aggs.views(*p) {
                    table.add_pre_agg(pre_agg, groups)?;
                }
                Ok(())
            },
            state,
        )?;
        pre_aggs.free(state)?;

        let params = &*self.params;
        let prepare: &PrepareFrame<'_> =
            &move |input, frame| params.prepare_frame(input, frame, state).boxed();
        let add_wave = |(p, table, _): &mut (usize, PartitionTable, DataFrame),
                        wave: &[LoadedFrame]|
         -> PolarsResult<()> {
            // Partitions start at different frames, so that tasks don't read the same frame at
            // the same time.
            let start = *p * wave.len() / num_partitions;
            for i in 0..wave.len() {
                let frame = &wave[(start + i) % wave.len()];
                let rows = frame.partition_rows(*p);
                if rows.is_some_and(|rows| rows.is_empty()) {
                    continue;
                }
                table.add_rows(frame.input, &frame.frame, rows)?;
            }
            Ok(())
        };
        build_in_rounds(
            waves,
            partitioner,
            &mut partitions,
            Some(prepare),
            &add_wave,
            state,
        )?;

        for_each_parallel(
            &mut partitions,
            &|(_, table, out): &mut (usize, PartitionTable, DataFrame)| -> PolarsResult<()> {
                *out = table.finish()?;
                Ok(())
            },
            state,
        )?;
        let mut outputs: Vec<DataFrame> = partitions.into_iter().map(|(_, _, out)| out).collect();
        // An empty result is still sent, as one empty morsel.
        if outputs.iter().any(|out| out.height() > 0) {
            outputs.retain(|out| out.height() > 0);
        } else {
            outputs.truncate(1);
        }
        Ok(outputs)
    }

    /// Writes the cold rows out as pieces per partition, so that the partitions can be built a
    /// few at a time while the results are sent.
    fn write_out(
        &self,
        waves: Vec<Vec<(usize, KeyedSpillFrame)>>,
        partitioner: &HashPartitioner,
        state: &StreamingExecutionState,
    ) -> PolarsResult<Vec<StoredPartition>> {
        let spill_ctx =
            MostRecentSpillContext::new("group-by-partition".into(), state.task_metrics.clone());
        let mut stored: Vec<StoredPartition> = (0..partitioner.num_partitions())
            .map(|p| StoredPartition::new(p, self.num_inputs, spill_ctx.clone()))
            .collect();
        build_in_rounds(
            waves,
            partitioner,
            &mut stored,
            None,
            &|stored: &mut StoredPartition, wave: &[LoadedFrame]| {
                stored.write_wave(wave);
                Ok(())
            },
            state,
        )?;
        Ok(stored)
    }
}

impl ComputeNode for GroupByNode {
    fn name(&self) -> &str {
        "group-by"
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, GroupByState::Sink(_))
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == self.num_inputs && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                self.state = GroupByState::Done;
            },
            // All inputs are done, build the results and transition to sending them.
            GroupByState::Sink(_) if recv.iter().all(|r| matches!(r, PortState::Done)) => {
                let GroupByState::Sink(sink) =
                    core::mem::replace(&mut self.state, GroupByState::Done)
                else {
                    unreachable!()
                };
                self.state = GroupByState::Emit(self.finish(sink, state)?);
            },
            GroupByState::Emit(emit) if emit.is_finished() => {
                self.state = GroupByState::Done;
            },
            // Nothing to change.
            GroupByState::Done | GroupByState::Sink(_) | GroupByState::Emit(_) => {},
        }

        // Communicate our state.
        match &self.state {
            GroupByState::Sink(_) => {
                recv.fill(PortState::Ready);
                send[0] = PortState::Blocked;
            },
            GroupByState::Emit(_) => {
                recv.fill(PortState::Done);
                send[0] = PortState::Ready;
            },
            GroupByState::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send_ports.len() == 1 && recv_ports.len() == self.num_inputs);
        match &mut self.state {
            GroupByState::Sink(sink) => {
                assert!(send_ports[0].is_none());
                assert!(recv_ports.iter().any(|r| r.is_some()));

                // If we have multiple input streams merge them into one (still identifying which
                // input stream it came from).
                let (senders, receivers): (Vec<_>, Vec<_>) =
                    (0..self.num_pipelines).map(|_| channel(1)).unzip();
                for (i, recv_port) in recv_ports.iter_mut().enumerate() {
                    if let Some(recv_port) = recv_port.take() {
                        for (mut r, s) in recv_port
                            .parallel()
                            .into_iter()
                            .zip(senders.iter().cloned())
                        {
                            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                                while let Ok(morsel) = r.recv().await {
                                    if s.send((i, morsel)).await.is_err() {
                                        break;
                                    }
                                }

                                Ok(())
                            }));
                        }
                    }
                }
                sink.spawn(
                    scope,
                    receivers,
                    state,
                    join_handles,
                    &self.params,
                    &self.spill_ctx,
                )
            },
            GroupByState::Emit(emit) => {
                assert!(recv_ports.iter().all(|r| r.is_none()));
                let senders = send_ports[0].take().unwrap().parallel();
                emit.spawn(scope, senders, state, join_handles);
            },
            GroupByState::Done => unreachable!(),
        }
    }
}
