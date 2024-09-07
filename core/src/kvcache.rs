use std::ops::Range;

use getset::{CopyGetters, Getters, MutGetters};
use num_traits::Num;

use crate::tensor::{Tensor, TensorView, WritableTensorView};

#[derive(CopyGetters)]
pub struct KVCache<TID: Copy, P: Num + Default + Copy> {
    blocks: Vec<KvCacheBlock<TID, P>>,
    #[getset(get_copy = "pub")]
    layer_num: usize,
    #[getset(get_copy = "pub")]
    dim: usize,
    #[getset(get_copy = "pub")]
    max_seq_len: usize,
}

impl<TID: Copy, P: Num + Default + Copy> KVCache<TID, P> {
    pub fn new(layer_num: usize, max_seq_len: usize, dim: usize) -> Self {
        KVCache {
            blocks: Vec::new(),
            layer_num,
            dim,
            max_seq_len,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn token_ids(&self) -> impl Iterator<Item = TID> + '_ {
        self.blocks.iter().map(|block| block.token_id)
    }

    /// Pushes a new token to the cache
    pub fn push(&mut self, token_id: TID) -> Result<(), &'static str> {
        if self.seq_len() >= self.max_seq_len {
            return Err("Exceeds max sequence length");
        }

        self.blocks
            .push(KvCacheBlock::new(token_id, self.layer_num, self.dim));
        Ok(())
    }

    pub fn push_from(&mut self, token_ids: &[TID]) -> Result<(), &'static str> {
        if self.seq_len() + token_ids.len() > self.max_seq_len {
            return Err("Exceeds max sequence length");
        }

        self.blocks.extend(
            token_ids
                .iter()
                .map(|&token_id| KvCacheBlock::new(token_id, self.layer_num, self.dim)),
        );
        Ok(())
    }

    pub fn pop(&mut self) -> Option<TID> {
        self.blocks.pop().map(|block| block.token_id)
    }

    pub fn truncate(&mut self, len: usize) {
        self.blocks.truncate(len);
    }

    pub fn k_cache_within(
        &self,
        layer_idx: usize,
        seq_range: Range<usize>,
    ) -> CachedTensor<'_, TID, P> {
        assert!(layer_idx < self.layer_num);
        assert!(seq_range.end <= self.seq_len());

        let range_len = seq_range.end - seq_range.start;
        CachedTensor {
            blocks: &self.blocks[seq_range.clone()],
            target: CacheTarget::K,
            length: range_len * self.dim,
            offset: 0,
            shape: vec![range_len, self.dim],
            layer_idx,
        }
    }
    pub fn v_cache_within(
        &self,
        layer_idx: usize,
        seq_range: Range<usize>,
    ) -> CachedTensor<'_, TID, P> {
        assert!(layer_idx < self.layer_num);
        assert!(seq_range.end <= self.seq_len());

        let range_len = seq_range.end - seq_range.start;
        CachedTensor {
            blocks: &self.blocks[seq_range.clone()],
            target: CacheTarget::V,
            length: range_len * self.dim,
            offset: 0,
            shape: vec![range_len, self.dim],
            layer_idx,
        }
    }

    pub fn full_k_cache(&self, layer_idx: usize) -> CachedTensor<'_, TID, P> {
        self.k_cache_within(layer_idx, 0..self.seq_len())
    }
    pub fn full_v_cache(&self, layer_idx: usize) -> CachedTensor<'_, TID, P> {
        self.v_cache_within(layer_idx, 0..self.seq_len())
    }
}

#[derive(Getters, MutGetters, CopyGetters)]
struct KvCacheBlock<TID: Copy, P: Num + Default + Copy> {
    #[getset(get_copy = "pub")]
    token_id: TID,
    #[getset(get = "pub", get_mut = "pub")]
    k: Vec<Tensor<P>>, // [n_kv_head * dqkv] * layers
    #[getset(get = "pub", get_mut = "pub")]
    v: Vec<Tensor<P>>, // [n_kv_head * dqkv] * layers
}

impl<T: Copy, P: Num + Default + Copy> KvCacheBlock<T, P> {
    pub fn new(token_id: T, layer_num: usize, dim: usize) -> Self {
        KvCacheBlock {
            token_id,
            k: (0..layer_num).map(|_| Tensor::default(&[dim])).collect(),
            v: (0..layer_num).map(|_| Tensor::default(&[dim])).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CacheTarget {
    K,
    V,
}
#[derive(Getters, CopyGetters)]
pub struct CachedTensor<'a, T: Copy, P: Num + Default + Copy> {
    blocks: &'a [KvCacheBlock<T, P>], // [sliced_seq_len, union{[n_kv_head * dqkv] * layers}]
    target: CacheTarget,
    length: usize,
    offset: usize,
    #[getset(get = "pub")]
    shape: Vec<usize>,
    #[getset(get_copy = "pub")]
    layer_idx: usize,
}

unsafe impl<'a, T: Copy, P: Num + Default + Copy> WritableTensorView<P> for CachedTensor<'a, T, P> {
    unsafe fn with_data_mut_at(&mut self, idx: &[usize], op: impl FnOnce(&P) -> P) -> P {
        let (target, offset) = self.to_target(idx);
        let target = target as *const Tensor<P> as *mut Tensor<P>;
        target.as_mut().unwrap().with_data_mut_at(&[offset], op)
    }
    unsafe fn data_iter_mut<'b>(&'b mut self) -> impl Iterator<Item = &'b mut P>
    where
        P: 'b,
    {
        self.data_iter()
            .map(|p| (p as *const P as *mut P).as_mut().unwrap())
    }
}

impl<'a, T: Copy, P: Num + Default + Copy> CachedTensor<'a, T, P> {
    pub fn reshape(&mut self, shape: &[usize]) -> &mut Self {
        assert_eq!(shape.iter().product::<usize>(), self.length);
        self.shape = shape.to_vec();
        self
    }

    fn to_target(&self, idx: &[usize]) -> (&Tensor<P>, usize) {
        let mut offset = self.to_offset(idx) + self.offset;
        let mut block_idx = 0;
        while offset >= self.blocks[block_idx].k[self.layer_idx].size() {
            offset -= self.blocks[block_idx].k[self.layer_idx].size();
            block_idx += 1;
        }
        let target = match self.target {
            CacheTarget::K => &self.blocks[block_idx].k[self.layer_idx],
            CacheTarget::V => &self.blocks[block_idx].v[self.layer_idx],
        };
        (target, offset)
    }
}

impl<'a, T: Copy, P: Num + Default + Copy> TensorView<P> for CachedTensor<'a, T, P> {
    fn data_at<'b>(&'b self, idx: &[usize]) -> &'b P {
        let (target, offset) = self.to_target(idx);
        target.data_at(&[offset])
    }

    fn data_iter<'b>(&'b self) -> impl Iterator<Item = &'b P>
    where
        P: 'b,
    {
        self.blocks
            .iter()
            .flat_map(|block| match self.target {
                CacheTarget::K => block.k[self.layer_idx].data_iter(),
                CacheTarget::V => block.v[self.layer_idx].data_iter(),
            })
            .skip(self.offset)
            .take(self.length)
    }

    fn size(&self) -> usize {
        self.length
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn slice(&self, start: usize, shape: &[usize]) -> Self {
        let length = shape.iter().product();
        assert!(length <= self.length && start <= self.length - length);
        let mut offset = self.offset + start;
        let mut block_idx = 0;
        while offset >= self.blocks[block_idx].k[self.layer_idx].size() {
            offset -= self.blocks[block_idx].k[self.layer_idx].size();
            block_idx += 1;
        }

        Self {
            blocks: &self.blocks[block_idx..],
            length,
            offset,
            shape: shape.to_vec(),
            ..*self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_cached_tensor {
        use super::*;

        fn default_test_cache_blocks<P: Num + Default + Copy>() -> Vec<KvCacheBlock<u32, P>> {
            let token_ids = (0..3).into_iter();
            token_ids.map(|i| KvCacheBlock::new(i, 2, 3)).collect()
        }

        #[test]
        fn test_read_cached_tensor() {
            let blocks = vec![
                KvCacheBlock {
                    token_id: 0,
                    k: vec![
                        Tensor::new(vec![0., 1., 2.], &[3]),
                        Tensor::new(vec![3., 4., 5.], &[3]),
                    ],
                    v: vec![
                        Tensor::new(vec![6., 7., 8.], &[3]),
                        Tensor::new(vec![9., 10., 11.], &[3]),
                    ],
                },
                KvCacheBlock {
                    token_id: 1,
                    k: vec![
                        Tensor::new(vec![12., 13., 14.], &[3]),
                        Tensor::new(vec![15., 16., 17.], &[3]),
                    ],
                    v: vec![
                        Tensor::new(vec![18., 19., 20.], &[3]),
                        Tensor::new(vec![21., 22., 23.], &[3]),
                    ],
                },
            ];
            {
                // read k
                let cached_tensor = CachedTensor {
                    blocks: &blocks,
                    target: CacheTarget::K,
                    length: 6,
                    offset: 0,
                    shape: vec![2, 3],
                    layer_idx: 0,
                };
                assert_eq!(cached_tensor.size(), 6);
                assert_eq!(cached_tensor.shape(), &[2, 3]);
                assert_eq!(*cached_tensor.data_at(&[0, 0]), 0.);
                assert_eq!(*cached_tensor.data_at(&[1, 2]), 14.);
                let iter_order = [0., 1., 2., 12., 13., 14.];
                cached_tensor
                    .data_iter()
                    .zip(iter_order)
                    .for_each(|(x, y)| {
                        assert_eq!(x, &y);
                    });

                let sliced_cached_tensor = cached_tensor.slice(2, &[2, 1]);
                assert_eq!(sliced_cached_tensor.size(), 2);
                assert_eq!(sliced_cached_tensor.shape(), &[2, 1]);
                assert_eq!(*sliced_cached_tensor.data_at(&[1, 0]), 12.);
            }
            {
                // read v
                let cached_tensor = CachedTensor {
                    blocks: &blocks[1..],
                    target: CacheTarget::V,
                    length: 3,
                    offset: 0,
                    shape: vec![1, 3],
                    layer_idx: 1,
                };
                assert_eq!(cached_tensor.size(), 3);
                assert_eq!(*cached_tensor.data_at(&[0, 1]), 22.);
            }
        }

        #[test]
        fn test_write_cached_tensor() {
            let mut blocks = default_test_cache_blocks::<f32>();
            let mut cached_tensor = CachedTensor {
                blocks: &mut blocks,
                target: CacheTarget::V,
                length: 6,
                offset: 3,
                shape: vec![2, 3],
                layer_idx: 1,
            };
            assert!(cached_tensor.data_iter().all(|&x| x == 0.));
            let new_vals = (0..cached_tensor.size())
                .map(|x| x as f32)
                .collect::<Vec<f32>>();
            unsafe {
                cached_tensor
                    .data_iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| {
                        *x = new_vals[i];
                    });
            }
            assert!(cached_tensor
                .data_iter()
                .zip(new_vals.iter())
                .all(|(x, y)| x == y));

            let mut sliced_cached_tensor = cached_tensor.slice(2, &[2, 2]);
            let idx = [1, 1];
            let prev_val = unsafe { sliced_cached_tensor.with_data_mut_at(&idx, |&x| x + 233.) };
            assert_eq!(prev_val, new_vals.last().unwrap().clone());
            assert!(
                (sliced_cached_tensor.data_at(&idx) - (new_vals.last().unwrap().clone() + 233.))
                    .abs()
                    <= 1e-6
            );
        }
    }

    mod test_kvcache {
        use super::*;

        #[test]
        fn test_create_kvcache() {
            let cache = KVCache::<u32, f32>::new(2, 3, 3);
            assert_eq!(cache.seq_len(), 0);
            assert_eq!(cache.layer_num(), 2);
            assert_eq!(cache.dim(), 3);
            assert_eq!(cache.max_seq_len(), 3);
        }

        #[test]
        fn test_push_kvcache() {
            let mut cache = KVCache::<u32, f32>::new(2, 3, 3);
            cache.push(0).unwrap();
            cache.push(1).unwrap();
            cache.push(2).unwrap();
            assert_eq!(cache.seq_len(), 3);
            assert_eq!(cache.token_ids().collect::<Vec<u32>>(), vec![0, 1, 2]);

            let tids = 0..2;
            let mut cache = KVCache::<u32, f32>::new(3, 2, 1);
            cache
                .push_from(tids.clone().collect::<Vec<_>>().as_slice())
                .unwrap();
            assert_eq!(cache.seq_len(), tids.len());
            assert!(cache.token_ids().zip(tids).all(|(x, y)| x == y));
        }

        #[test]
        fn test_ranged_kv_cache() {
            let tids = 0..2;
            let mut cache = KVCache::<u32, f32>::new(1, 2, 1);
            cache
                .push_from(tids.clone().collect::<Vec<_>>().as_slice())
                .unwrap();

            let cached_tensor = cache.full_k_cache(0);
            assert_eq!(cached_tensor.size(), 2);
            assert_eq!(cached_tensor.shape(), &[2, 1]);

            let cached_tensor = cache.k_cache_within(0, 0..1);
            assert_eq!(cached_tensor.size(), 1);
            assert_eq!(cached_tensor.shape(), &[1, 1]);
        }

        #[test]
        #[should_panic]
        fn test_push_out_of_capacity() {
            let mut cache = KVCache::<u32, f32>::new(1, 3, 1);
            cache.push(0).unwrap();
            cache.push(1).unwrap();
            cache.push(2).unwrap();
            cache.push(3).unwrap(); // panic!
        }

        #[test]
        #[should_panic]
        fn test_layer_out_of_range() {
            let cache = KVCache::<u32, f32>::new(1, 3, 1);
            cache.full_k_cache(1); // panic!
        }
    }
}
