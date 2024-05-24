use arrow::bitmap::MutableBitmap;
use arrow::legacy::kernels::set::set_at_nulls;
use arrow::legacy::trusted_len::FromIteratorReversed;
use arrow::legacy::utils::FromTrustedLenIterator;
use bytemuck::Zeroable;
use num_traits::{Bounded, NumCast, One, Zero};

use crate::prelude::*;

fn err_fill_null() -> PolarsError {
    polars_err!(ComputeError: "could not determine the fill value")
}

impl Series {
    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    /// * Zero fill (replace None with the value zero)
    /// * One fill (replace None with the value one)
    /// * MinBound fill (replace with the minimum of that data type)
    /// * MaxBound fill (replace with the maximum of that data type)
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`fill_null` operation on `ChunkedArray<T>`](crate::chunked_array::ops::ChunkFillNullValue)*.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn example() -> PolarsResult<()> {
    ///     let s = Series::new("some_missing", &[Some(1), None, Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Forward(None))?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Backward(None))?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Min)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Max)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Mean)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::Zero)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(0), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::One)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::MinBound)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(-2147483648), Some(2)]);
    ///
    ///     let filled = s.fill_null(FillNullStrategy::MaxBound)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2147483647), Some(2)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Series> {
        let logical_type = self.dtype();
        let s = self.to_physical_repr();

        use DataType::*;
        let out = match s.dtype() {
            Boolean => fill_null_bool(s.bool().unwrap(), strategy),
            String => {
                let s = unsafe { s.cast_unchecked(&Binary)? };
                let out = s.fill_null(strategy)?;
                return unsafe { out.cast_unchecked(&String) };
            },
            Binary => {
                let ca = s.binary().unwrap();
                fill_null_binary(ca, strategy).map(|ca| ca.into_series())
            },
            List(_) => {
                let ca = s.list().unwrap();
                fill_null_list(ca, strategy).map(|ca| ca.into_series())
            },
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                        fill_null_numeric(ca, strategy).map(|ca| ca.into_series())
                })
            },
            _ => todo!(),
        }?;
        unsafe { out.cast_unchecked(logical_type) }
    }
}

// Utility trait to make generics work
trait LocalCopy {
    fn cheap_clone(&self) -> Self;
}

impl<T: Copy> LocalCopy for T {
    #[inline]
    fn cheap_clone(&self) -> Self {
        *self
    }
}

impl LocalCopy for Series {
    #[inline]
    fn cheap_clone(&self) -> Self {
        self.clone()
    }
}

fn fill_forward_limit<'a, T, K, I>(ca: &'a ChunkedArray<T>, limit: IdxSize) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: LocalCopy,
    I: TrustedLen<Item = Option<K>>,
    T: PolarsDataType,
    ChunkedArray<T>: FromTrustedLenIterator<Option<K>>,
{
    let mut cnt = 0;
    let mut previous = None;
    ca.into_iter()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v.cheap_clone());
                Some(v)
            },
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous.as_ref().map(|v| v.cheap_clone())
                } else {
                    None
                }
            },
        })
        .collect_trusted()
}

fn fill_backward_limit<'a, T, K, I>(ca: &'a ChunkedArray<T>, limit: IdxSize) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: LocalCopy,
    I: TrustedLen<Item = Option<K>> + DoubleEndedIterator,
    T: PolarsDataType,
    ChunkedArray<T>: FromIteratorReversed<Option<K>>,
{
    let mut cnt = 0;
    let mut previous = None;
    ca.into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v.cheap_clone());
                Some(v)
            },
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous.as_ref().map(|v| v.cheap_clone())
                } else {
                    None
                }
            },
        })
        .collect_reversed()
}

fn fill_backward_limit_binary(ca: &BinaryChunked, limit: IdxSize) -> BinaryChunked {
    let mut cnt = 0;
    let mut previous = None;
    let out: BinaryChunked = ca
        .into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v);
                Some(v)
            },
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous
                } else {
                    None
                }
            },
        })
        .collect_trusted();
    out.into_iter().rev().collect_trusted()
}

fn fill_forward<'a, T, K, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: LocalCopy,
    I: TrustedLen<Item = Option<K>>,
    T: PolarsDataType,
    ChunkedArray<T>: FromTrustedLenIterator<Option<K>>,
{
    ca.into_iter()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value.cheap_clone());
                Some(Some(value))
            },
            None => Some(previous.as_ref().map(|v| v.cheap_clone())),
        })
        .collect_trusted()
}

fn fill_backward<'a, T, K, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    K: Copy,
    I: TrustedLen<Item = Option<K>> + DoubleEndedIterator,
    T: PolarsDataType,
    ChunkedArray<T>: FromIteratorReversed<Option<K>>,
{
    ca.into_iter()
        .rev()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value);
                Some(Some(value))
            },
            None => Some(*previous),
        })
        .collect_reversed()
}

macro_rules! impl_fill_backward {
    ($ca:ident, $ChunkedArray:ty) => {{
        let ca: $ChunkedArray = $ca
            .into_iter()
            .rev()
            .scan(None, |previous, opt_v| match opt_v {
                Some(value) => {
                    *previous = Some(value.cheap_clone());
                    Some(Some(value))
                },
                None => Some(previous.as_ref().map(|s| s.cheap_clone())),
            })
            .collect_trusted();
        ca.into_iter().rev().collect_trusted()
    }};
}

macro_rules! impl_fill_backward_limit {
    ($ca:ident, $ChunkedArray:ty, $limit:expr) => {{
        let mut cnt = 0;
        let mut previous = None;
        let out: $ChunkedArray = $ca
            .into_iter()
            .rev()
            .map(|opt_v| match opt_v {
                Some(v) => {
                    cnt = 0;
                    previous = Some(v.cheap_clone());
                    Some(v)
                },
                None => {
                    if cnt < $limit {
                        cnt += 1;
                        previous.as_ref().map(|s| s.cheap_clone())
                    } else {
                        None
                    }
                },
            })
            .collect_trusted();
        out.into_iter().rev().collect_trusted()
    }};
}

fn fill_forward_numeric<'a, T, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    I: TrustedLen + Iterator<Item = Option<T::Physical<'a>>>,
    T::ZeroablePhysical<'a>: LocalCopy,
{
    // Compute values.
    let values: Vec<T::ZeroablePhysical<'a>> = ca
        .into_iter()
        .scan(T::ZeroablePhysical::zeroed(), |prev, v| {
            *prev = v.map(|v| v.into()).unwrap_or(prev.cheap_clone());
            Some(prev.cheap_clone())
        })
        .collect_trusted();

    // Compute bitmask.
    let num_start_nulls = ca.first_non_null().unwrap_or(0);
    let mut bm = MutableBitmap::with_capacity(ca.len());
    bm.extend_constant(num_start_nulls, false);
    bm.extend_constant(ca.len() - num_start_nulls, true);
    ChunkedArray::from_chunk_iter_like(
        ca,
        [
            T::Array::from_zeroable_vec(values, ca.dtype().to_arrow(true))
                .with_validity_typed(Some(bm.into())),
        ],
    )
}

fn fill_backward_numeric<'a, T, I>(ca: &'a ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator<IntoIter = I>,
    I: TrustedLen + Iterator<Item = Option<T::Physical<'a>>> + DoubleEndedIterator,
    T::ZeroablePhysical<'a>: LocalCopy,
{
    // Compute values.
    let values: Vec<T::ZeroablePhysical<'a>> = ca
        .into_iter()
        .rev()
        .scan(T::ZeroablePhysical::zeroed(), |prev, v| {
            *prev = v.map(|v| v.into()).unwrap_or(prev.cheap_clone());
            Some(prev.cheap_clone())
        })
        .collect_reversed();

    // Compute bitmask.
    let last_idx = ca.len().saturating_sub(1);
    let num_end_nulls = last_idx - ca.last_non_null().unwrap_or(last_idx);
    let mut bm = MutableBitmap::with_capacity(ca.len());
    bm.extend_constant(ca.len() - num_end_nulls, true);
    bm.extend_constant(num_end_nulls, false);
    ChunkedArray::from_chunk_iter_like(
        ca,
        [
            T::Array::from_zeroable_vec(values, ca.dtype().to_arrow(true))
                .with_validity_typed(Some(bm.into())),
        ],
    )
}

fn fill_null_numeric<T>(
    ca: &ChunkedArray<T>,
    strategy: FillNullStrategy,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    // Nothing to fill.
    if ca.null_count() == 0 {
        return Ok(ca.clone());
    }
    let mut out = match strategy {
        FillNullStrategy::Forward(None) => fill_forward_numeric(ca),
        FillNullStrategy::Forward(Some(limit)) => fill_forward_limit(ca, limit),
        FillNullStrategy::Backward(None) => fill_backward_numeric(ca),
        FillNullStrategy::Backward(Some(limit)) => fill_backward_limit(ca, limit),
        FillNullStrategy::Min => {
            ca.fill_null_with_values(ChunkAgg::min(ca).ok_or_else(err_fill_null)?)?
        },
        FillNullStrategy::Max => {
            ca.fill_null_with_values(ChunkAgg::max(ca).ok_or_else(err_fill_null)?)?
        },
        FillNullStrategy::Mean => ca.fill_null_with_values(
            ca.mean()
                .map(|v| NumCast::from(v).unwrap())
                .ok_or_else(err_fill_null)?,
        )?,
        FillNullStrategy::One => return ca.fill_null_with_values(One::one()),
        FillNullStrategy::Zero => return ca.fill_null_with_values(Zero::zero()),
        FillNullStrategy::MinBound => return ca.fill_null_with_values(Bounded::min_value()),
        FillNullStrategy::MaxBound => return ca.fill_null_with_values(Bounded::max_value()),
    };
    out.rename(ca.name());
    Ok(out)
}

fn fill_null_bool(ca: &BooleanChunked, strategy: FillNullStrategy) -> PolarsResult<Series> {
    // Nothing to fill.
    if ca.null_count() == 0 {
        return Ok(ca.clone().into_series());
    }
    match strategy {
        FillNullStrategy::Forward(limit) => {
            let mut out: BooleanChunked = match limit {
                Some(limit) => fill_forward_limit(ca, limit),
                None => fill_forward(ca),
            };
            out.rename(ca.name());
            Ok(out.into_series())
        },
        FillNullStrategy::Backward(limit) => {
            let mut out: BooleanChunked = match limit {
                None => fill_backward(ca),
                Some(limit) => fill_backward_limit(ca, limit),
            };
            out.rename(ca.name());
            Ok(out.into_series())
        },
        FillNullStrategy::Min => ca
            .fill_null_with_values(ca.min().ok_or_else(err_fill_null)?)
            .map(|ca| ca.into_series()),
        FillNullStrategy::Max => ca
            .fill_null_with_values(ca.max().ok_or_else(err_fill_null)?)
            .map(|ca| ca.into_series()),
        FillNullStrategy::Mean => polars_bail!(opq = mean, "Boolean"),
        FillNullStrategy::One | FillNullStrategy::MaxBound => {
            ca.fill_null_with_values(true).map(|ca| ca.into_series())
        },
        FillNullStrategy::Zero | FillNullStrategy::MinBound => {
            ca.fill_null_with_values(false).map(|ca| ca.into_series())
        },
    }
}

fn fill_null_binary(ca: &BinaryChunked, strategy: FillNullStrategy) -> PolarsResult<BinaryChunked> {
    // Nothing to fill.
    if ca.null_count() == 0 {
        return Ok(ca.clone());
    }
    match strategy {
        FillNullStrategy::Forward(limit) => {
            let mut out: BinaryChunked = match limit {
                Some(limit) => fill_forward_limit(ca, limit),
                None => fill_forward(ca),
            };
            out.rename(ca.name());
            Ok(out)
        },
        FillNullStrategy::Backward(limit) => {
            let mut out = match limit {
                None => impl_fill_backward!(ca, BinaryChunked),
                Some(limit) => fill_backward_limit_binary(ca, limit),
            };
            out.rename(ca.name());
            Ok(out)
        },
        FillNullStrategy::Min => {
            ca.fill_null_with_values(ca.min_binary().ok_or_else(err_fill_null)?)
        },
        FillNullStrategy::Max => {
            ca.fill_null_with_values(ca.max_binary().ok_or_else(err_fill_null)?)
        },
        FillNullStrategy::Zero => ca.fill_null_with_values(&[]),
        strat => polars_bail!(InvalidOperation: "fill-null strategy {:?} is not supported", strat),
    }
}

fn fill_null_list(ca: &ListChunked, strategy: FillNullStrategy) -> PolarsResult<ListChunked> {
    // Nothing to fill.
    if ca.null_count() == 0 {
        return Ok(ca.clone());
    }
    match strategy {
        FillNullStrategy::Forward(limit) => {
            let mut out: ListChunked = match limit {
                Some(limit) => fill_forward_limit(ca, limit),
                None => fill_forward(ca),
            };
            out.rename(ca.name());
            Ok(out)
        },
        FillNullStrategy::Backward(limit) => {
            let mut out: ListChunked = match limit {
                None => impl_fill_backward!(ca, ListChunked),
                Some(limit) => impl_fill_backward_limit!(ca, ListChunked, limit),
            };
            out.rename(ca.name());
            Ok(out)
        },
        strat => polars_bail!(InvalidOperation: "fill-null strategy {:?} is not supported", strat),
    }
}

impl<T> ChunkFillNullValue<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn fill_null_with_values(&self, value: T::Native) -> PolarsResult<Self> {
        Ok(self.apply_kernel(&|arr| Box::new(set_at_nulls(arr, value))))
    }
}

impl ChunkFillNullValue<bool> for BooleanChunked {
    fn fill_null_with_values(&self, value: bool) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNullValue<&[u8]> for BinaryChunked {
    fn fill_null_with_values(&self, value: &[u8]) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}
