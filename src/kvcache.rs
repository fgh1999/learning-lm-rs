use getset::{Getters, MutGetters};
use num_traits::Num;

use crate::tensor::Tensor;

#[derive(Getters, MutGetters)]
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize, // n_kv_head * dqkv: length of one key/value tensor
    #[getset(get = "pub", get_mut = "pub")]
    seq_len: usize, // length of the current sequence
}

impl<T: Num + Default + Copy> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, seq_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&[max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&[max_seq_len, dim]))
                .collect(),
            max_seq_len,
            dim,
            seq_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &[self.seq_len - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &[self.seq_len - start, self.dim])
    }
}
