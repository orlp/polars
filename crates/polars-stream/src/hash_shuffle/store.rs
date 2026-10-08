use polars_ooc::{MostRecentSpillContext, ParameterFreeSpillContext};

use super::keyed_frame::{KeyedFrameBuilder, KeyedSpillFrame};
use super::rounds::LoadedFrame;

/// The pieces of one partition, per input, one per round.
pub(crate) struct StoredPartition {
    index: usize,
    pieces_per_input: Vec<Vec<KeyedSpillFrame>>,
    estimated_size: u64,
    /// Keeps the pieces registered, so they stay spillable while the partition exists.
    spill_ctx: MostRecentSpillContext,
}

impl StoredPartition {
    pub(crate) fn new(index: usize, num_inputs: usize, spill_ctx: MostRecentSpillContext) -> Self {
        Self {
            index,
            pieces_per_input: (0..num_inputs).map(|_| Vec::new()).collect(),
            estimated_size: 0,
            spill_ctx,
        }
    }

    pub(crate) fn index(&self) -> usize {
        self.index
    }

    /// The estimated memory of the pieces, in bytes.
    pub(crate) fn estimated_size(&self) -> u64 {
        self.estimated_size
    }

    /// Gathers this partition's rows of every frame of `wave` by index into one piece per input:
    /// reserved exactly, registered for spilling without waiting. Empty lists are skipped. Each
    /// input's columns are those of its first frame in the wave. The frames must be split.
    pub(crate) fn write_wave(&mut self, wave: &[LoadedFrame]) {
        let p = self.index;
        for input in 0..self.pieces_per_input.len() {
            let rows_per_frame = || {
                wave.iter()
                    .filter(move |f| f.input == input)
                    .map(move |f| (f, f.partition_rows(p).expect("frames must be split")))
            };
            let num_rows: usize = rows_per_frame().map(|(_, rows)| rows.len()).sum();
            if num_rows == 0 {
                continue;
            }

            let (first, _) = rows_per_frame().next().unwrap();
            let schema = first.frame.frame.schema().clone();
            let mut builder = KeyedFrameBuilder::new(schema, &first.frame.keys);
            builder.reserve(num_rows);
            for (f, rows) in rows_per_frame() {
                if !rows.is_empty() {
                    // SAFETY: the rows are rows of this frame, whose columns are single chunks
                    // once loaded.
                    unsafe { builder.gather_extend(&f.frame.frame, &f.frame.keys, rows) };
                }
            }

            let piece = KeyedSpillFrame::new(builder.freeze_reset());
            self.spill_ctx.register_no_spill_check(&piece);
            self.estimated_size += piece.estimated_size() as u64;
            self.pieces_per_input[input].push(piece);
        }
    }

    /// The next piece, input by input.
    pub(crate) fn pop_piece(&mut self) -> Option<(usize, KeyedSpillFrame)> {
        self.pieces_per_input
            .iter_mut()
            .enumerate()
            .find_map(|(input, pieces)| Some((input, pieces.pop()?)))
    }
}
