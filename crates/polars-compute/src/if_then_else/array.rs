use arrow::array::growable::{Growable, GrowableFixedSizeList};
use arrow::array::{Array, ArrayCollectIterExt, FixedSizeListArray};
use arrow::bitmap::utils::SlicesIterator;
use arrow::bitmap::Bitmap;

use super::IfThenElseKernel;

impl IfThenElseKernel for FixedSizeListArray {
    type Scalar<'a> = Box<dyn Array>;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let mut growable = GrowableFixedSizeList::new(vec![if_true, if_false], false, mask.len());
        unsafe {
            if_then_else_extend(
                &mut growable,
                mask,
                |g, off, len| g.extend(0, off, len),
                |g, off, len| g.extend(1, off, len),
            )
        };
        growable.to()
    }

    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self {
        let if_true_list: FixedSizeListArray =
            std::iter::once(if_true).collect_arr_trusted_with_dtype(if_false.data_type().clone());
        let mut growable = GrowableFixedSizeList::new(vec![&if_true_list, if_false], false, mask.len());
        unsafe {
            if_then_else_extend(
                &mut growable,
                mask,
                |g, _, len| g.extend_copies(0, 0, 1, len),
                |g, off, len| g.extend(1, off, len),
            )
        };
        growable.to()
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let if_false_list: FixedSizeListArray =
            std::iter::once(if_false).collect_arr_trusted_with_dtype(if_true.data_type().clone());
        let mut growable = GrowableFixedSizeList::new(vec![if_true, &if_false_list], false, mask.len());
        unsafe {
            if_then_else_extend(
                &mut growable,
                mask,
                |g, off, len| g.extend(0, off, len),
                |g, _, len| g.extend_copies(1, 0, 1, len),
            )
        };
        growable.to()
    }

    fn if_then_else_broadcast_both(
        dtype: arrow::datatypes::ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let if_true_list: FixedSizeListArray =
            std::iter::once(if_true).collect_arr_trusted_with_dtype(dtype.clone());
        let if_false_list: FixedSizeListArray =
            std::iter::once(if_false).collect_arr_trusted_with_dtype(dtype.clone());
        let mut growable = GrowableFixedSizeList::new(vec![&if_true_list, &if_false_list], false, mask.len());
        unsafe {
            if_then_else_extend(
                &mut growable,
                mask,
                |g, _, len| g.extend_copies(0, 0, 1, len),
                |g, _, len| g.extend_copies(1, 0, 1, len),
            )
        };
        growable.to()
    }
}

fn if_then_else_extend<G, ET: Fn(&mut G, usize, usize), EF: Fn(&mut G, usize, usize)>(
    growable: &mut G,
    mask: &Bitmap,
    extend_true: ET,
    extend_false: EF,
) {
    let mut last_true_end = 0;
    for (start, len) in SlicesIterator::new(mask) {
        if start != last_true_end {
            extend_false(growable, last_true_end, start - last_true_end);
        };
        extend_true(growable, start, len);
        last_true_end = start + len;
    }
    if last_true_end != mask.len() {
        extend_false(growable, last_true_end, mask.len() - last_true_end)
    }
}
