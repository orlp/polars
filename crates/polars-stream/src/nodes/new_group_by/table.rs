use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::IntoColumn;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::groups::Grouper;
use polars_expr::reduce::GroupedReduction;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::pre_agg::PreAgg;
use crate::execute::StreamingExecutionState;
use crate::hash_shuffle::keyed_frame::KeyedFrame;
use crate::nodes::group_by::InputPayload;

/// What the sink and every partition table read.
pub(super) struct GroupByParams {
    pub(super) key_schema: Arc<Schema>,
    pub(super) output_schema: Arc<Schema>,
    /// Input stream i feeds grouped_reductions[k] for each k in reductions_per_input[i].
    pub(super) reductions_per_input: Vec<Vec<usize>>,
    /// grouped_reductions[k] is passed input cols grouped_reduction_cols[k].
    pub(super) grouped_reduction_cols: Vec<Vec<PlSmallStr>>,
    pub(super) payload_per_input: Vec<InputPayload>,
    /// Templates for the grouper and reductions of each partition.
    pub(super) grouper: Box<dyn Grouper>,
    pub(super) grouped_reductions: Vec<Box<dyn GroupedReduction>>,
    pub(super) has_order_sensitive_agg: bool,
}

impl GroupByParams {
    /// Adds the fused columns of the input to a frame of its cold rows, evaluated once on all
    /// its rows.
    pub(super) async fn prepare_frame(
        &self,
        input: usize,
        mut frame: KeyedFrame,
        state: &StreamingExecutionState,
    ) -> PolarsResult<KeyedFrame> {
        for selector in &self.payload_per_input[input].fused_selectors {
            let c = selector
                .evaluate_preserve_len_broadcast(&frame.frame, &state.in_memory_exec_state)
                .await?;
            unsafe { frame.frame.push_column_unchecked(c.rechunk()) };
        }
        Ok(frame)
    }
}

/// `rows`, or else the first `len` row indices, from `all_rows`.
fn rows_or_all<'a>(
    rows: Option<&'a [IdxSize]>,
    all_rows: &'a mut Vec<IdxSize>,
    len: usize,
) -> &'a [IdxSize] {
    match rows {
        Some(rows) => rows,
        None => {
            if all_rows.len() < len {
                all_rows.extend(all_rows.len() as IdxSize..len as IdxSize);
            }
            &all_rows[..len]
        },
    }
}

/// The groups and reduction states of one partition.
pub(super) struct PartitionTable {
    params: Arc<GroupByParams>,
    grouper: Box<dyn Grouper>,
    reductions: Vec<Box<dyn GroupedReduction>>,
    group_idxs: Vec<IdxSize>,
    all_rows: Vec<IdxSize>,
}

impl PartitionTable {
    pub(super) fn new(params: Arc<GroupByParams>, reserve: usize) -> Self {
        let mut grouper = params.grouper.new_empty();
        grouper.reserve(reserve);
        let reductions = params
            .grouped_reductions
            .iter()
            .map(|r| {
                let mut r = r.new_empty();
                r.reserve(reserve);
                r
            })
            .collect();
        Self {
            params,
            grouper,
            reductions,
            group_idxs: Vec::new(),
            all_rows: Vec::new(),
        }
    }

    /// Combines the groups `groups` (all if `None`) of a pre-aggregate into this table.
    pub(super) fn add_pre_agg(
        &mut self,
        pre_agg: &PreAgg,
        groups: Option<&[IdxSize]>,
    ) -> PolarsResult<()> {
        let groups = rows_or_all(groups, &mut self.all_rows, pre_agg.keys.len());
        self.group_idxs.clear();
        unsafe {
            self.grouper
                .insert_keys_subset(&pre_agg.keys, groups, Some(&mut self.group_idxs));
        }
        for (pre_reduction, r_idx) in pre_agg.reductions.iter().zip(pre_agg.reduction_idxs.iter()) {
            let r = &mut self.reductions[*r_idx];
            r.resize(self.grouper.num_groups());
            unsafe { r.combine_subset(&**pre_reduction, groups, &self.group_idxs)? };
        }
        Ok(())
    }

    /// Inserts the rows `rows` (all if `None`) of a prepared frame of `input` into this table.
    pub(super) fn add_rows(
        &mut self,
        input: usize,
        frame: &KeyedFrame,
        rows: Option<&[IdxSize]>,
    ) -> PolarsResult<()> {
        let rows = rows_or_all(rows, &mut self.all_rows, frame.keys.len());
        self.group_idxs.clear();
        unsafe {
            self.grouper
                .insert_keys_subset(&frame.keys, rows, Some(&mut self.group_idxs));
        }
        let num_groups = self.grouper.num_groups();
        let params = &*self.params;
        let mut in_cols = Vec::new();
        for r_idx in &params.reductions_per_input[input] {
            let r = &mut self.reductions[*r_idx];
            r.resize(num_groups);
            in_cols.clear();
            in_cols.extend(
                params.grouped_reduction_cols[*r_idx]
                    .iter()
                    .map(|col| frame.frame.column(col).unwrap()),
            );
            // Cold rows only exist without order-sensitive reductions.
            unsafe { r.update_groups_subset(&in_cols, rows, &self.group_idxs, 0)? };
        }
        Ok(())
    }

    /// The keys in group order followed by the finalized reductions. Leaves the table empty.
    pub(super) fn finish(&mut self) -> PolarsResult<DataFrame> {
        let grouper = std::mem::replace(&mut self.grouper, self.params.grouper.new_empty());
        let num_groups = grouper.num_groups();
        let mut out = grouper.get_keys_in_group_order(&self.params.key_schema);
        drop(grouper);
        let out_names = self.params.output_schema.iter_names().skip(out.width());
        for (r, name) in self.reductions.iter_mut().zip(out_names) {
            // Each input only resizes its own reductions.
            r.resize(num_groups);
            let c = r.finalize()?.with_name(name.clone()).into_column();
            unsafe { out.push_column_unchecked(c) };
        }
        Ok(out)
    }
}
