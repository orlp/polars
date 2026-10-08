use futures::future::join;
use polars_arrow::array::BooleanArray;
use polars_arrow::array::builder::ShareStrategy;
use polars_arrow::bitmap::Bitmap;
use polars_core::datatypes::{ArrowDataType, BooleanChunked};
use polars_core::frame::DataFrame;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::IntoColumn;
use polars_core::schema::SchemaRef;
use polars_expr::hash_keys::{HashKeys, HashKeysBuilder, SpilledHashKeys};
use polars_ooc::{SpillToken, Spillable};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

/// Rows with their hashed keys, and an optional mask of the rows that belong to it.
pub(crate) struct KeyedFrame {
    pub(crate) frame: DataFrame,
    pub(crate) keys: HashKeys,
    pub(crate) mask: Option<Bitmap>,
}

impl KeyedFrame {
    /// The number of rows that belong to this frame.
    fn num_rows(&self) -> usize {
        self.mask.as_ref().map_or(self.keys.len(), |m| m.set_bits())
    }

    fn estimated_size(&self) -> usize {
        self.frame.estimated_size(false)
            + self.keys.estimated_size()
            + self.mask.as_ref().map_or(0, |m| m.len().div_ceil(8))
    }
}

/// A spilled `KeyedFrame`: the frame, with the mask as one more column if there is one, and the
/// keys.
pub(crate) struct SpilledKeyedFrame {
    frame: <DataFrame as Spillable>::Spilled,
    keys: SpilledHashKeys,
    mask_column: Option<PlSmallStr>,
}

impl Spillable for KeyedFrame {
    type Spilled = SpilledKeyedFrame;

    fn estimate_byte_size(&self) -> usize {
        self.estimated_size()
    }

    async fn spill(&self, context_id: &str) -> SpilledKeyedFrame {
        let mut frame = self.frame.clone();
        let mask_column = self.mask.as_ref().map(|mask| {
            let mut name = String::from("mask");
            while frame.schema().contains(&name) {
                name.push('_');
            }
            let name = PlSmallStr::from_string(name);
            let mask = BooleanArray::new(ArrowDataType::Boolean, mask.clone(), None);
            let column = BooleanChunked::with_chunk(name.clone(), mask).into_column();
            frame.with_column(column).unwrap();
            name
        });
        let (frame, keys) = join(frame.spill(context_id), self.keys.spill(context_id)).await;
        SpilledKeyedFrame {
            frame,
            keys,
            mask_column,
        }
    }

    async fn unspill(spilled: &SpilledKeyedFrame) -> Self {
        let (mut frame, keys) = join(
            DataFrame::unspill(&spilled.frame),
            HashKeys::unspill(&spilled.keys),
        )
        .await;
        let mask = spilled.mask_column.as_ref().map(|name| {
            let column = frame.drop_in_place(name).unwrap();
            let mask = column.bool().unwrap().rechunk();
            mask.downcast_as_array().values().clone()
        });
        if frame.width() == 0 {
            frame = DataFrame::empty_with_height(keys.len());
        }
        Self { frame, keys, mask }
    }
}

/// A keyed frame that can spill: one `SpillToken`, its size known while spilled.
pub(crate) struct KeyedSpillFrame {
    token: SpillToken<KeyedFrame>,
    num_rows: usize,
    estimated_size: usize,
}

impl AsRef<SpillToken<KeyedFrame>> for KeyedSpillFrame {
    fn as_ref(&self) -> &SpillToken<KeyedFrame> {
        &self.token
    }
}

impl KeyedSpillFrame {
    /// Not registered with any spill context.
    pub(crate) fn new(frame: KeyedFrame) -> Self {
        Self {
            num_rows: frame.num_rows(),
            estimated_size: frame.estimated_size(),
            token: SpillToken::new(frame),
        }
    }

    /// The number of rows that belong to the frame: those set in its mask, or all rows.
    pub(crate) fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The estimated memory of the frame, in bytes.
    pub(crate) fn estimated_size(&self) -> usize {
        self.estimated_size
    }

    /// Whether the frame is in memory and not being spilled.
    pub(crate) fn is_in_memory(&self) -> bool {
        self.token.try_get().is_some()
    }

    /// Takes the frame out of its token, unspilling it if needed, with every column in one
    /// chunk.
    pub(crate) async fn load(self) -> KeyedFrame {
        let mut frame = self.token.into_inner().await;
        frame.frame.rechunk_mut();
        frame
    }
}

/// Appends row subsets of keyed frames of one input; strings are copied (`ShareStrategy::Never`).
pub(crate) struct KeyedFrameBuilder {
    frame: DataFrameBuilder,
    keys: HashKeysBuilder,
}

impl KeyedFrameBuilder {
    /// A builder for frames of `schema` with keys of the kind and key schema of `like`.
    pub(crate) fn new(schema: SchemaRef, like: &HashKeys) -> Self {
        Self {
            frame: DataFrameBuilder::new(schema),
            keys: HashKeysBuilder::new(like),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        self.frame.reserve(additional);
        self.keys.reserve(additional);
    }

    /// # Safety
    /// The rows must be in-bounds of `frame` and `keys`, which have the same number of rows; the
    /// columns of `frame` must be single chunks.
    pub(crate) unsafe fn gather_extend(
        &mut self,
        frame: &DataFrame,
        keys: &HashKeys,
        rows: &[IdxSize],
    ) {
        unsafe {
            self.frame.gather_extend(frame, rows, ShareStrategy::Never);
            self.keys.gather_extend(keys, rows);
        }
    }

    /// Takes the appended rows, leaving the builder empty.
    pub(crate) fn freeze_reset(&mut self) -> KeyedFrame {
        KeyedFrame {
            frame: self.frame.freeze_reset(),
            keys: self.keys.freeze_reset(),
            mask: None,
        }
    }
}
