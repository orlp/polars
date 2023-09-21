//! Contains the concatenate kernel
//!
//! Example:
//!
//! ```
//! use arrow2::array::Utf8Array;
//! use arrow2::compute::concatenate::concatenate;
//!
//! let arr = concatenate(&[
//!     &Utf8Array::<i32>::from_slice(["hello", "world"]),
//!     &Utf8Array::<i32>::from_slice(["!"]),
//! ]).unwrap();
//! assert_eq!(arr.len(), 3);
//! ```

use crate::array::growable::make_growable;
use crate::array::Array;
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::error::{Error, Result};

/// Concatenate multiple [Array] of the same type into a single [`Array`].
pub fn concatenate(arrays: &[&dyn Array]) -> Result<Box<dyn Array>> {
    if arrays.is_empty() {
        return Err(Error::InvalidArgumentError(
            "concat requires input of at least one array".to_string(),
        ));
    }

    if arrays
        .iter()
        .any(|array| array.data_type() != arrays[0].data_type())
    {
        return Err(Error::InvalidArgumentError(
            "It is not possible to concatenate arrays of different data types.".to_string(),
        ));
    }

    let lengths = arrays.iter().map(|array| array.len()).collect::<Vec<_>>();
    let capacity = lengths.iter().sum();

    let mut mutable = make_growable(arrays, false, capacity);

    for (i, len) in lengths.iter().enumerate() {
        mutable.extend(i, 0, *len)
    }

    Ok(mutable.as_box())
}


/// Concatenate the validities of multiple [Array]s into a single Bitmap.
pub fn concatenate_validities(arrays: &[&dyn Array]) -> Option<Bitmap> {
    let null_count: usize = arrays.iter().map(|a| a.null_count()).sum();
    if null_count == 0 {
        return None;
    }

    let total_size: usize = arrays.iter().map(|a| a.len()).sum();
    let mut bitmap = MutableBitmap::with_capacity(total_size);
    for arr in arrays {
        if arr.null_count() == arr.len() {
            bitmap.extend_constant(arr.len(), false);
        } else if arr.null_count() == 0 {
            bitmap.extend_constant(arr.len(), true);
        } else {
            bitmap.extend_from_bitmap(arr.validity().unwrap());
        }
    }
    Some(bitmap.into())
}
