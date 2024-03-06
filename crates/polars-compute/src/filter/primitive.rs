use arrow::bitmap::Bitmap;
use bytemuck::{cast_slice, cast_vec, Pod};

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use super::avx512;

use super::boolean::filter_boolean_kernel;
use super::scalar::{scalar_filter, scalar_filter_offset};

fn nop_filter<'a, T: Pod>(values: &'a [T], mask: &'a [u8], out: *mut T) -> (&'a [T], &'a [u8], *mut T) {
    (values, mask, out)
}

pub fn filter_values<T: Pod>(values: &[T], mask: &Bitmap) -> Vec<T> {
    match (std::mem::size_of::<T>(), std::mem::align_of::<T>()) {
        (1, 1) => cast_vec(filter_values_u8(cast_slice(values), mask)),
        (2, 2) => cast_vec(filter_values_u16(cast_slice(values), mask)),
        (4, 4) => cast_vec(filter_values_u32(cast_slice(values), mask)),
        (8, 8) => cast_vec(filter_values_u64(cast_slice(values), mask)),
        (16, _) => filter_values_u128(values, mask),
        _ => filter_values_generic(values, mask, 1, &nop_filter),
    }
}

fn filter_values_u8(values: &[u8], mask: &Bitmap) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512vbmi2") {
        unsafe {
            return filter_values_generic(values, mask, 64, &|v, m, o| avx512::filter_u8_avx512vbmi2(v, m, o));
        }
    }

    filter_values_generic(values, mask, 1, &nop_filter)
}

fn filter_values_u16(values: &[u16], mask: &Bitmap) -> Vec<u16> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512vbmi2") {
        unsafe {
            return filter_values_generic(values, mask, 32, &|v, m, o| avx512::filter_u16_avx512vbmi2(v, m, o));
        }
    }

    filter_values_generic(values, mask, 1, &nop_filter)
}

fn filter_values_u32(values: &[u32], mask: &Bitmap) -> Vec<u32> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            return filter_values_generic(values, mask, 16, &|v, m, o| avx512::filter_u32_avx512f(v, m, o));
        }
    }

    filter_values_generic(values, mask, 1, &nop_filter)
}

fn filter_values_u64(values: &[u64], mask: &Bitmap) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            return filter_values_generic(values, mask, 8, &|v, m, o| avx512::filter_u64_avx512f(v, m, o));
        }
    }

    filter_values_generic(values, mask, 1, &nop_filter)
}

fn filter_values_u128<T: Pod>(values: &[T], mask: &Bitmap) -> Vec<T> {
    assert_eq!(std::mem::size_of::<T>(), 16);

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            return filter_values_generic(values, mask, 4, &|v, m, o| {
                let v = std::mem::transmute::<&[T], &[[u8; 16]]>(v);
                let (v, m, o) = avx512::filter_u128_unaligned_avx512f(v, m, o as *mut [u8; 16]);
                let v = std::mem::transmute::<&[[u8; 16]], &[T]>(v);
                (v, m, o as *mut T)
            });
        }
    }

    filter_values_generic(values, mask, 1, &nop_filter)
}

fn filter_values_generic<T: Pod>(
    values: &[T],
    mask: &Bitmap,
    pad: usize,
    bulk_filter: &dyn for<'a> Fn(&'a [T], &'a [u8], *mut T) -> (&'a [T], &'a [u8], *mut T),
) -> Vec<T> {
    assert_eq!(values.len(), mask.len());
    let mask_bits_set = mask.set_bits();
    let mut out = Vec::with_capacity(mask_bits_set + pad);
    unsafe {
        let (values, mask_bytes, out_ptr) = scalar_filter_offset(values, mask, out.as_mut_ptr());
        let (values, mask_bytes, out_ptr) = bulk_filter(values, mask_bytes, out_ptr);
        scalar_filter(values, mask_bytes, out_ptr);
        out.set_len(mask_bits_set);
    }
    out
}

pub fn filter_values_and_validity<T: Pod>(
    values: &[T],
    validity: Option<&Bitmap>,
    mask: &Bitmap,
) -> (Vec<T>, Option<Bitmap>) {
    (
        filter_values(values, mask),
        validity.map(|v| filter_boolean_kernel(v, mask)),
    )
}
