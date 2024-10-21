use std::fmt::Debug;

use num_traits::{FromPrimitive, ToPrimitive};
use polars_utils::idx_vec::IdxVec;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::sync::SyncPtr;
use rayon::prelude::*;

#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
use crate::config::verbose;
use crate::datatypes::*;
use crate::prelude::*;
use crate::POOL;

impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: ToPrimitive + FromPrimitive + Debug,
{
    // Use the indexes as perfect groups
    pub fn group_tuples_perfect(
        &self,
        max: usize,
        mut multithreaded: bool,
        group_capacity: usize,
    ) -> GroupsProxy {
        multithreaded &= POOL.current_num_threads() > 1;
        let len = if self.null_count() > 0 {
            // we add one to store the null sentinel group
            max + 2
        } else {
            max + 1
        };

        // the latest index will be used for the null sentinel
        let null_idx = len.saturating_sub(1);
        let n_threads = POOL.current_num_threads();

        let chunk_size = len / n_threads;

        let (groups, first) = if multithreaded && chunk_size > 1 {
            let mut groups: Vec<IdxVec> = Vec::new();
            groups.resize_with(len, || IdxVec::with_capacity(group_capacity));
            let mut first: Vec<IdxSize> = Vec::with_capacity(len);

            // Round up offsets to nearest cache line for groups to reduce false sharing.
            let groups_start = groups.as_ptr();
            let mut per_thread_offsets = Vec::with_capacity(n_threads + 1);
            per_thread_offsets.push(0);
            for t in 0..n_threads {
                let ideal_offset = (t + 1) * chunk_size;
                let cache_aligned_offset =
                    ideal_offset + groups_start.wrapping_add(ideal_offset).align_offset(128);
                per_thread_offsets.push(cache_aligned_offset);
            }

            let groups_ptr = unsafe { SyncPtr::new(groups.as_mut_ptr()) };
            let first_ptr = unsafe { SyncPtr::new(first.as_mut_ptr()) };
            POOL.install(|| {
                (0..n_threads).into_par_iter().for_each(|thread_no| {
                    // We use raw pointers because the slices would overlap.
                    // However, each thread has its own range it is responsible for.
                    let groups = groups_ptr.get();
                    let first = first_ptr.get();
                    let start = per_thread_offsets[thread_no];
                    let start = T::Native::from_usize(start).unwrap();
                    let end = per_thread_offsets[thread_no + 1];
                    let end = T::Native::from_usize(end).unwrap();

                    let mut row_nr = 0 as IdxSize;
                    for arr in self.downcast_iter() {
                        if arr.null_count() == 0 {
                            for &cat in arr.values().as_slice() {
                                if cat >= start && cat < end {
                                    let cat = cat.to_usize().unwrap();
                                    let buf = unsafe { &mut *groups.add(cat) };
                                    buf.push(row_nr);

                                    unsafe {
                                        if buf.len() == 1 {
                                            // SAFETY: we just  pushed
                                            let first_value = buf.get_unchecked(0);
                                            *first.add(cat) = *first_value;
                                        }
                                    }
                                }
                                row_nr += 1;
                            }
                        } else {
                            for opt_cat in arr.iter() {
                                if let Some(&cat) = opt_cat {
                                    // cannot factor out due to bchk
                                    if cat >= start && cat < end {
                                        let cat = cat.to_usize().unwrap();
                                        let buf = unsafe { &mut *groups.add(cat) };
                                        buf.push(row_nr);

                                        unsafe {
                                            if buf.len() == 1 {
                                                // SAFETY: we just  pushed
                                                let first_value = buf.get_unchecked(0);
                                                *first.add(cat) = *first_value;
                                            }
                                        }
                                    }
                                }
                                // last thread handles null values
                                else if thread_no == n_threads - 1 {
                                    let buf = unsafe { &mut *groups.add(null_idx) };
                                    buf.push(row_nr);
                                    unsafe {
                                        if buf.len() == 1 {
                                            let first_value = buf.get_unchecked(0);
                                            *first.add(null_idx) = *first_value;
                                        }
                                    }
                                }

                                row_nr += 1;
                            }
                        }
                    }
                });
            });
            unsafe {
                first.set_len(len);
            }
            (groups, first)
        } else {
            let mut groups = Vec::with_capacity(len);
            let mut first = vec![IdxSize::MAX; len];
            groups.resize_with(len, || IdxVec::with_capacity(group_capacity));

            let mut row_nr = 0 as IdxSize;
            for arr in self.downcast_iter() {
                for opt_cat in arr.iter() {
                    if let Some(cat) = opt_cat {
                        let group_id = cat.to_usize().unwrap();
                        let buf = unsafe { groups.get_unchecked_release_mut(group_id) };
                        buf.push(row_nr);

                        unsafe {
                            if buf.len() == 1 {
                                *first.get_unchecked_release_mut(group_id) = row_nr;
                            }
                        }
                    } else {
                        let buf = unsafe { groups.get_unchecked_release_mut(null_idx) };
                        buf.push(row_nr);
                        unsafe {
                            let first_value = buf.get_unchecked(0);
                            *first.get_unchecked_release_mut(null_idx) = *first_value
                        }
                    }

                    row_nr += 1;
                }
            }
            (groups, first)
        };

        // NOTE! we set sorted here!
        // this happens to be true for `fast_unique` categoricals
        GroupsProxy::Idx(GroupsIdx::new(first, groups, true))
    }
}

#[cfg(all(feature = "dtype-categorical", feature = "performant"))]
// Special implementation so that cats can be processed in a single pass
impl CategoricalChunked {
    // Use the indexes as perfect groups
    pub fn group_tuples_perfect(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        let rev_map = self.get_rev_map();
        if self.is_empty() {
            return GroupsProxy::Idx(GroupsIdx::new(vec![], vec![], true));
        }
        let cats = self.physical();

        let mut out = match &**rev_map {
            RevMapping::Local(cached, _) => {
                if self._can_fast_unique() {
                    if verbose() {
                        eprintln!("grouping categoricals, run perfect hash function");
                    }
                    // on relative small tables this isn't much faster than the default strategy
                    // but on huge tables, this can be > 2x faster
                    cats.group_tuples_perfect(cached.len() - 1, multithreaded, 0)
                } else {
                    self.physical().group_tuples(multithreaded, sorted).unwrap()
                }
            },
            RevMapping::Global(_mapping, _cached, _) => {
                // TODO! see if we can optimize this
                // the problem is that the global categories are not guaranteed packed together
                // so we might need to deref them first to local ones, but that might be more
                // expensive than just hashing (benchmark first)
                self.physical().group_tuples(multithreaded, sorted).unwrap()
            },
        };
        if sorted {
            out.sort()
        }
        out
    }
}
