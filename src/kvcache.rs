use std::ops::Range;

use getset::{Getters, MutGetters};
use num_traits::Num;

use crate::tensor::{Tensor, TensorView, WritableTensorView};

#[derive(Getters)]
pub struct KVCache<TID: Num + Copy, P: Num + Default + Copy> {
    blocks: Vec<KvCacheBlock<TID, P>>,
    #[getset(get = "pub")]
    layer_num: usize,
    #[getset(get = "pub")]
    dim: usize,
    #[getset(get = "pub")]
    max_seq_len: usize,
}

impl<TID: Num + Copy, P: Num + Default + Copy> KVCache<TID, P> {
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

    pub fn token_ids(&self) -> Vec<TID> {
        self.blocks.iter().map(|block| block.token_id).collect()
    }

    /// Pushes a new token to the cache, returns the index of the new cache block
    pub fn push(&mut self, token_id: TID) -> Result<usize, ()> {
        if self.seq_len() >= self.max_seq_len {
            return Err(());
        }
        self.blocks
            .push(KvCacheBlock::new(token_id, self.layer_num, self.dim));
        Ok(self.seq_len() - 1)
    }

    pub fn k_cache_within<'a>(
        &'a self,
        layer_idx: usize,
        seq_range: Range<usize>,
    ) -> CachedTensor<'a, TID, P> {
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
    pub fn v_cache_within<'a>(
        &'a self,
        layer_idx: usize,
        seq_range: Range<usize>,
    ) -> CachedTensor<'a, TID, P> {
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

    pub fn full_k_cache<'a>(&'a self, layer_idx: usize) -> CachedTensor<'a, TID, P> {
        self.k_cache_within(layer_idx, 0..self.seq_len())
    }
    pub fn full_v_cache<'a>(&'a self, layer_idx: usize) -> CachedTensor<'a, TID, P> {
        self.v_cache_within(layer_idx, 0..self.seq_len())
    }
}

#[derive(Getters, MutGetters)]
struct KvCacheBlock<TID: Num + Copy, P: Num + Default + Copy> {
    #[getset(get = "pub")]
    token_id: TID,
    #[getset(get = "pub", get_mut = "pub")]
    k: Vec<Tensor<P>>, // [n_kv_head * dqkv] * layers
    #[getset(get = "pub", get_mut = "pub")]
    v: Vec<Tensor<P>>, // [n_kv_head * dqkv] * layers
}

impl<T: Num + Copy, P: Num + Default + Copy> KvCacheBlock<T, P> {
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
#[derive(Getters)]
pub struct CachedTensor<'a, T: Num + Copy, P: Num + Default + Copy> {
    blocks: &'a [KvCacheBlock<T, P>], // [sliced_seq_len, union{[n_kv_head * dqkv] * layers}]
    target: CacheTarget,
    length: usize,
    offset: usize,
    #[getset(get = "pub")]
    shape: Vec<usize>,
    #[getset(get = "pub")]
    layer_idx: usize,
}

unsafe impl<'a, T: Num + Copy, P: Num + Default + Copy> WritableTensorView<P>
    for CachedTensor<'a, T, P>
{
    unsafe fn with_data_mut_at(&mut self, idx: &[usize], op: impl FnOnce(&P) -> P) -> P {
        let mut offset = self.to_offset(idx);
        let mut block_idx = 0;
        while offset >= self.blocks[block_idx].k[self.layer_idx].size() {
            offset -= self.blocks[block_idx].k[self.layer_idx].size();
            block_idx += 1;
        }
        let target = {
            match self.target {
                CacheTarget::K => &self.blocks[block_idx].k[self.layer_idx],
                CacheTarget::V => &self.blocks[block_idx].v[self.layer_idx],
            }
        };
        unsafe {
            let target = target as *const Tensor<P> as *mut Tensor<P>;
            target.as_mut().unwrap().with_data_mut_at(&[offset], op)
        }
    }
    unsafe fn data_iter_mut<'b>(&'b mut self) -> impl Iterator<Item = &'b mut P>
    where
        P: 'b,
    {
        self.data_iter()
            .map(|p| (p as *const P as *mut P).as_mut().unwrap())
    }
}

impl<'a, T: Num + Copy, P: Num + Default + Copy> CachedTensor<'a, T, P> {
    pub fn reshape(&mut self, shape: &[usize]) -> &mut Self {
        assert_eq!(shape.iter().product::<usize>(), self.length);
        self.shape = shape.to_vec();
        self
    }
}

impl<'a, T: Num + Copy, P: Num + Default + Copy> TensorView<P> for CachedTensor<'a, T, P> {
    fn data_at(&self, idx: &[usize]) -> P {
        let mut offset = self.to_offset(idx) + self.offset;
        let mut block_idx = 0;
        while offset >= self.blocks[block_idx].k[self.layer_idx].size() {
            offset -= self.blocks[block_idx].k[self.layer_idx].size();
            block_idx += 1;
        }
        match self.target {
            CacheTarget::K => self.blocks[block_idx].k[self.layer_idx].data_at(&[offset]),
            CacheTarget::V => self.blocks[block_idx].v[self.layer_idx].data_at(&[offset]),
        }
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
