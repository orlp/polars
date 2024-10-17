use hashbrown::hash_table::{Entry, HashTable};
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_unordered;
use polars_utils::aliases::PlRandomState;
use polars_utils::itertools::Itertools;
use polars_utils::vec::PushUnchecked;

use super::*;

struct Group {
    key_hash: u64,
    key_offset: usize,
    key_length: u32,
    group_idx: IdxSize,
}

impl Group {
    unsafe fn key<'s, 'k>(&'s self, key_data: &'k [u8]) -> &'k [u8] {
        key_data.get_unchecked(self.key_offset..self.key_offset + self.key_length as usize)
    }
}

#[derive(Default)]
pub struct RowEncodedHashGrouper {
    key_schema: Arc<Schema>,
    table: HashTable<Group>,
    key_data: Vec<u8>,
    random_state: PlRandomState,
}

impl RowEncodedHashGrouper {
    pub fn new(key_schema: Arc<Schema>, random_state: PlRandomState) -> Self {
        Self {
            key_schema,
            random_state,
            ..Default::default()
        }
    }

    fn insert_key(&mut self, hash: u64, key: &[u8]) -> IdxSize {
        let num_groups = self.table.len();
        let entry = self.table.entry(
            hash,
            |g| unsafe { hash == g.key_hash && key == g.key(&self.key_data) },
            |g| g.key_hash,
        );

        match entry {
            Entry::Occupied(e) => e.get().group_idx,
            Entry::Vacant(e) => {
                let group_idx: IdxSize = num_groups.try_into().unwrap();
                let group = Group {
                    key_hash: hash,
                    key_offset: self.key_data.len(),
                    key_length: key.len().try_into().unwrap(),
                    group_idx,
                };
                self.key_data.extend(key);
                e.insert(group);
                group_idx
            },
        }
    }

    fn finalize_keys(&self, mut key_rows: Vec<&[u8]>) -> DataFrame {
        let key_dtypes = self
            .key_schema
            .iter()
            .map(|(_name, dt)| dt.to_physical().to_arrow(CompatLevel::newest()))
            .collect::<Vec<_>>();
        let fields = vec![Default::default(); key_dtypes.len()];
        let key_columns =
            unsafe { polars_row::decode::decode_rows(&mut key_rows, &fields, &key_dtypes) };

        let cols = self
            .key_schema
            .iter()
            .zip(key_columns)
            .map(|((name, _dt), col)| Series::try_from((name.clone(), col)).unwrap().into_column())
            .collect();
        unsafe { DataFrame::new_no_checks_height_from_first(cols) }
    }
}

impl Grouper for RowEncodedHashGrouper {
    fn new_empty(&self) -> Box<dyn Grouper> {
        Box::new(Self::new(
            self.key_schema.clone(),
            self.random_state.clone(),
        ))
    }

    fn num_groups(&self) -> IdxSize {
        self.table.len() as IdxSize
    }

    fn insert_keys(&mut self, keys: &[Column], group_idxs: &mut Vec<IdxSize>) {
        let series = keys
            .iter()
            .map(|c| c.as_materialized_series().clone())
            .collect_vec();
        let keys_encoded = _get_rows_encoded_unordered(&series[..])
            .unwrap()
            .into_array();
        assert!(keys_encoded.len() == keys.len());

        group_idxs.clear();
        group_idxs.reserve(keys.len());
        for key in keys_encoded.values_iter() {
            let hash = self.random_state.hash_one(key);
            unsafe {
                group_idxs.push_unchecked(self.insert_key(hash, key));
            }
        }
    }

    fn combine(&mut self, other: &dyn Grouper, group_idxs: &mut Vec<IdxSize>) {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        group_idxs.clear();
        group_idxs.reserve(other.table.len());
        self.table.reserve(other.table.len(), |g| g.key_hash); // TODO: cardinality estimation.
        for group in other.table.iter() {
            unsafe {
                let group_key = group.key(&other.key_data);
                group_idxs.push_unchecked(self.insert_key(group.key_hash, group_key));
            }
        }
    }

    fn get_keys_in_group_order(&self) -> DataFrame {
        let mut key_rows: Vec<&[u8]> = Vec::with_capacity(self.table.len());
        unsafe {
            let p = key_rows.as_mut_ptr();
            for group in &self.table {
                *p.add(group.group_idx as usize) = group.key(&self.key_data);
            }
            key_rows.set_len(self.table.len());
        }
        self.finalize_keys(key_rows)
    }

    fn get_keys_groups(&self, group_idxs: &mut Vec<IdxSize>) -> DataFrame {
        group_idxs.clear();
        group_idxs.reserve(self.table.len());
        self.finalize_keys(
            self.table
                .iter()
                .map(|group| unsafe {
                    group_idxs.push(group.group_idx);
                    group.key(&self.key_data)
                })
                .collect(),
        )
    }

    fn partition_into(
        &self,
        _seed: u64,
        _partitions: &mut [Box<dyn Grouper>],
        _partition_idxs: &mut Vec<IdxSize>,
        _group_idxs: &mut Vec<IdxSize>,
    ) {
        unimplemented!()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
