use crossbeam_queue::ArrayQueue;
use futures::future::BoxFuture;
use polars_async::executor::{self, TaskPriority};
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_ooc::memory_manager;
use polars_utils::IdxSize;
use polars_utils::hashing::HashPartitioner;

use super::keyed_frame::{KeyedFrame, KeyedSpillFrame};
use super::split::{SplitRows, split};
use crate::execute::StreamingExecutionState;

/// Orders frames (those still in memory first, then by input) and cuts them into waves of about
/// `wave_bytes`, at least one frame each.
pub(crate) fn make_waves(
    frames: Vec<(usize, KeyedSpillFrame)>,
    wave_bytes: usize,
) -> Vec<Vec<(usize, KeyedSpillFrame)>> {
    let mut frames: Vec<_> = frames
        .into_iter()
        .map(|(input, frame)| (!frame.is_in_memory(), input, frame))
        .collect();
    frames.sort_by_key(|(spilled, input, _)| (*spilled, *input));

    let mut waves = Vec::new();
    let mut wave = Vec::new();
    let mut wave_size = 0;
    for (_, input, frame) in frames {
        if !wave.is_empty() && wave_size + frame.estimated_size() > wave_bytes {
            waves.push(std::mem::take(&mut wave));
            wave_size = 0;
        }
        wave_size += frame.estimated_size();
        wave.push((input, frame));
    }
    if !wave.is_empty() {
        waves.push(wave);
    }
    waves
}

/// A frame of the current wave: loaded, prepared and split.
pub(crate) struct LoadedFrame {
    pub(crate) input: usize,
    pub(crate) frame: KeyedFrame,
    /// `None`: all rows, in partition 0 (one partition, no mask).
    pub(crate) split: Option<SplitRows>,
}

impl LoadedFrame {
    /// The rows of partition `p`; `None` for all rows.
    pub(crate) fn partition_rows(&self, p: usize) -> Option<&[IdxSize]> {
        self.split.as_ref().map(|split| split.partition(p))
    }
}

/// Runs once on each loaded frame of the given input, before it is split.
pub(crate) type PrepareFrame<'p> =
    dyn Fn(usize, KeyedFrame) -> BoxFuture<'p, PolarsResult<KeyedFrame>> + Sync + 'p;

/// Calls `f` on every item, with up to `num_pipelines` tasks that take the items in slice order.
pub(crate) fn for_each_parallel<S: Send>(
    items: &mut [S],
    f: &(dyn Fn(&mut S) -> PolarsResult<()> + Sync),
    state: &StreamingExecutionState,
) -> PolarsResult<()> {
    if items.is_empty() {
        return Ok(());
    }
    let num_tasks = state.num_pipelines.min(items.len());
    let queue = queue_of(items.iter_mut());
    executor::task_scope(state.task_metrics(), |scope| {
        let handles: Vec<_> = (0..num_tasks)
            .map(|_| scope.spawn_task(TaskPriority::High, take_items(&queue, f)))
            .collect();
        ASYNC.block_in_place_on(async {
            for handle in handles {
                handle.await?;
            }
            PolarsResult::Ok(())
        })
    })
}

/// One round per wave, as `for_each_parallel` over the partitions: up to `num_pipelines` build
/// tasks (low priority) call `build` on each partition with the loaded wave, while high-priority
/// load tasks load, prepare and split the next wave (each frame once) and free the previous one.
pub(crate) fn build_in_rounds<S: Send>(
    waves: Vec<Vec<(usize, KeyedSpillFrame)>>,
    partitioner: &HashPartitioner,
    partitions: &mut [S],
    prepare: Option<&PrepareFrame<'_>>,
    build: &(dyn Fn(&mut S, &[LoadedFrame]) -> PolarsResult<()> + Sync),
    state: &StreamingExecutionState,
) -> PolarsResult<()> {
    let mut waves = waves.into_iter();
    let (mut current, mut done) = (Vec::new(), Vec::new());
    loop {
        let to_load = waves.next().unwrap_or_default();
        if to_load.is_empty() && current.is_empty() && done.is_empty() {
            return Ok(());
        }
        let loaded = run_round(
            to_load,
            done,
            &current,
            partitions,
            partitioner,
            prepare,
            build,
            state,
        )?;
        done = std::mem::replace(&mut current, loaded);
    }
}

/// One task scope: high-priority tasks drop the frames `to_free`, then load, prepare and split
/// the frames `to_load`, while low-priority tasks build every partition from `wave`, unless it
/// is empty. Returns the loaded frames.
#[allow(clippy::too_many_arguments)]
fn run_round<S: Send>(
    to_load: Vec<(usize, KeyedSpillFrame)>,
    to_free: Vec<LoadedFrame>,
    wave: &[LoadedFrame],
    partitions: &mut [S],
    partitioner: &HashPartitioner,
    prepare: Option<&PrepareFrame<'_>>,
    build: &(dyn Fn(&mut S, &[LoadedFrame]) -> PolarsResult<()> + Sync),
    state: &StreamingExecutionState,
) -> PolarsResult<Vec<LoadedFrame>> {
    let num_pipelines = state.num_pipelines;
    let num_load_tasks = num_pipelines.min(to_load.len().max(to_free.len()));
    let num_build_tasks = if wave.is_empty() {
        0
    } else {
        num_pipelines.min(partitions.len())
    };
    let load_queue = queue_of(to_load.into_iter());
    let free_queue = queue_of(to_free.into_iter());
    let partition_queue = queue_of(partitions.iter_mut());

    executor::task_scope(state.task_metrics(), |scope| {
        let (load_queue, free_queue) = (&load_queue, &free_queue);
        let load_handles: Vec<_> = (0..num_load_tasks)
            .map(|_| {
                scope.spawn_task(TaskPriority::High, async move {
                    while let Some(frame) = free_queue.pop() {
                        drop(frame);
                    }
                    let mut loaded = Vec::new();
                    while let Some((input, frame)) = load_queue.pop() {
                        loaded.push(load_frame(input, frame, partitioner, prepare).await?);
                        memory_manager().spill().await;
                    }
                    PolarsResult::Ok(loaded)
                })
            })
            .collect();

        let build_wave = move |partition: &mut S| build(partition, wave);
        let build_handles: Vec<_> = (0..num_build_tasks)
            .map(|_| scope.spawn_task(TaskPriority::Low, take_items(&partition_queue, build_wave)))
            .collect();

        ASYNC.block_in_place_on(async {
            let mut loaded = Vec::new();
            for handle in load_handles {
                loaded.extend(handle.await?);
            }
            for handle in build_handles {
                handle.await?;
            }
            PolarsResult::Ok(loaded)
        })
    })
}

async fn load_frame(
    input: usize,
    frame: KeyedSpillFrame,
    partitioner: &HashPartitioner,
    prepare: Option<&PrepareFrame<'_>>,
) -> PolarsResult<LoadedFrame> {
    let mut frame = frame.load().await;
    if let Some(prepare) = prepare {
        frame = prepare(input, frame).await?;
    }
    let split = split(&frame.keys, frame.mask.as_ref(), partitioner);
    Ok(LoadedFrame {
        input,
        frame,
        split,
    })
}

/// Calls `f` on items taken from `queue` until it is empty, letting the memory manager spill
/// after each item.
async fn take_items<T>(
    queue: &ArrayQueue<T>,
    f: impl Fn(T) -> PolarsResult<()>,
) -> PolarsResult<()> {
    while let Some(item) = queue.pop() {
        f(item)?;
        memory_manager().spill().await;
    }
    Ok(())
}

fn queue_of<T>(items: impl ExactSizeIterator<Item = T>) -> ArrayQueue<T> {
    let queue = ArrayQueue::new(items.len().max(1));
    for item in items {
        queue.push(item).ok().unwrap();
    }
    queue
}
