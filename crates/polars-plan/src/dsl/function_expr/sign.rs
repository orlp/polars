use polars_core::export::num;
use polars_core::with_match_physical_numeric_polars_type;
use num::{Zero, One};

use super::*;

pub(super) fn sign(s: &Series) -> PolarsResult<Series> {
    let dt = s.dtype();
    polars_ensure!(dt.is_numeric(), opq = sign, dt);
    with_match_physical_numeric_polars_type!(dt, |$T| {
        let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
        Ok(sign_impl(ca))
    })
}

fn sign_impl<T>(ca: &ChunkedArray<T>) -> Series
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries
{
    ca.apply_values(|x| {
        if x < T::Native::zero() {
            T::Native::zero() - T::Native::one()
        } else if x > T::Native::zero() {
            T::Native::one()
        } else {
            // Returning x here ensures we return NaN for NaN input, and
            // maintain the sign for signed zeroes (although we don't really
            // care about the latter).
            x
        }
    }).into_series()
}
