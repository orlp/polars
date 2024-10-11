use std::sync::Arc;

use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::Column;
use polars_core::schema::{Schema, SchemaExt};
use polars_expr::reduce::GroupedReduction;

use super::compute_node_prelude::*;
use crate::expression::StreamExpr;
use crate::morsel::SourceToken;
use crate::nodes::in_memory_source::InMemorySourceNode;

enum ReduceState {
    Sink {
        key_selectors: Vec<StreamExpr>, // Elementwise.
        grouped_reduction_selectors: Vec<StreamExpr>, // Elementwise.
        grouped_reductions: Vec<Box<dyn GroupedReduction>>,
    },
    Source(InMemorySourceNode),
    Done,
}

pub struct GroupByNode {
    state: ReduceState,
    output_schema: Arc<Schema>,
}

impl GroupByNode {
    pub fn new(
        key_selectors: Vec<StreamExpr>,
        grouped_reduction_selectors: Vec<StreamExpr>,
        grouped_reductions: Vec<Box<dyn GroupedReduction>>,
        output_schema: Arc<Schema>,
    ) -> Self {
        Self {
            state: ReduceState::Sink {
                key_selectors,
                grouped_reduction_selectors,
                grouped_reductions,
            },
            output_schema,
        }
    }

    fn spawn_sink<'env, 's>(
        selectors: &'env [StreamExpr],
        key_selectors: &'env [StreamExpr],
        grouped_reduction_selectors: &'env [StreamExpr],
        grouped_reductions: &'env mut [Box<dyn GroupedReduction>],
        scope: &'s TaskScope<'s, 'env>,
        recv: RecvPort<'_>,
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let parallel_tasks: Vec<_> = recv
            .parallel()
            .into_iter()
            .map(|mut recv| {
                let mut local_reducers: Vec<_> = reductions
                    .iter()
                    .map(|d| d.new_empty())
                    .collect();

                scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(morsel) = recv.recv().await {
                        for (reducer, selector) in local_reducers.iter_mut().zip(selectors) {
                            let input = selector.evaluate(morsel.df(), state).await?;
                            reducer.update_group(&input, 0)?;
                        }
                    }

                    PolarsResult::Ok(local_reducers)
                })
            })
            .collect();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            for task in parallel_tasks {
                let local_reducers = task.await?;
                for (r1, r2) in reductions.iter_mut().zip(local_reducers) {
                    r1.resize(1);
                    unsafe {
                        r1.combine(&*r2, &[0])?;
                    }
                }
            }

            Ok(())
        }));
    }

    fn spawn_source<'env, 's>(
        df: &'env mut Option<DataFrame>,
        scope: &'s TaskScope<'s, 'env>,
        send: SendPort<'_>,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let mut send = send.serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let morsel = Morsel::new(df.take().unwrap(), MorselSeq::new(0), SourceToken::new());
            let _ = send.send(morsel).await;
            Ok(())
        }));
    }
}

impl ComputeNode for GroupByNode {
    fn name(&self) -> &str {
        "reduce"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // State transitions.
        match &mut self.state {
            // If the output doesn't want any more data, transition to being done.
            _ if send[0] == PortState::Done => {
                self.state = ReduceState::Done;
            },
            // Input is done, transition to being a source.
            ReduceState::Sink { reductions, .. } if matches!(recv[0], PortState::Done) => {
                let columns = reductions
                    .iter_mut()
                    .zip(self.output_schema.iter_fields())
                    .map(|(r, field)| {
                        r.finalize().map(|s| {
                            let s = s.with_name(field.name.clone()).cast(&field.dtype).unwrap();
                            Column::Scalar(ScalarColumn::unit_scalar_from_series(s))
                        })
                    })
                    .try_collect_vec()?;
                let out = DataFrame::new(columns).unwrap();

                self.state = ReduceState::Source(Some(out));
            },
            // We have sent the reduced dataframe, we are done.
            ReduceState::Source(df) if df.is_none() => {
                self.state = ReduceState::Done;
            },
            // Nothing to change.
            ReduceState::Done | ReduceState::Sink { .. } | ReduceState::Source(_) => {},
        }

        // Communicate our state.
        match &self.state {
            ReduceState::Sink { .. } => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
            },
            ReduceState::Source(..) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
            },
            ReduceState::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send.len() == 1 && recv.len() == 1);
        match &mut self.state {
            ReduceState::Sink {
                selectors,
                reductions,
            } => {
                assert!(send[0].is_none());
                let recv_port = recv[0].take().unwrap();
                Self::spawn_sink(selectors, reductions, scope, recv_port, state, join_handles)
            },
            ReduceState::Source(df) => {
                assert!(recv[0].is_none());
                let send_port = send[0].take().unwrap();
                Self::spawn_source(df, scope, send_port, join_handles)
            },
            ReduceState::Done => unreachable!(),
        }
    }
}