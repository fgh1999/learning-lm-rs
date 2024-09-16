use std::ops::Range;

use crate::tensor::{Tensor, TensorIndex, TensorView};
use getset::{CopyGetters, Getters, MutGetters};
use num_traits::Num;

// #[derive(CopyGetters, Clone)]
// pub struct KvCacheUnit<P: Num> {
//     k: Tensor<P>, // [layers * n_kv_head * dqkv]
//     v: Tensor<P>, // [layers * n_kv_head * dqkv]
// }
// impl<P: Num + Default + Copy> KvCacheUnit<P> {
//     pub fn default(layer_num: usize, dim: usize) -> Self {
//         KvCacheUnit {
//             k: Tensor::default(&[layer_num, dim]),
//             v: Tensor::default(&[layer_num, dim]),
//         }
//     }
// }
// impl<P: Num> KvCacheUnit<P> {
//     /// Returns the k cache tensor of the specified layer (which is of shape: [dim])
//     fn k_on_layer(&self, layer_idx: usize) -> Tensor<P> {
//         self.k.slice(
//             self.k.to_offset(&[layer_idx, 0]),
//             &[self.k.shape().last().cloned().unwrap()],
//         )
//     }
//     /// Returns the v cache tensor of the specified layer (which is of shape: [dim])
//     fn v_on_layer(&self, layer_idx: usize) -> Tensor<P> {
//         self.v.slice(
//             self.v.to_offset(&[layer_idx, 0]),
//             &[self.v.shape().last().cloned().unwrap()],
//         )
//     }

//     /// Operates on the k cache tensor of the specified layer (which is of shape: [dim])
//     pub(in crate::kvcache_new) fn with_k_on_layer<R, F: FnOnce(&Tensor<P>) -> R>(&self, layer_idx: usize, op: F) -> R {
//         let k = self.k_on_layer(layer_idx);
//         op(&k)
//     }
//     /// Operates on the v cache tensor of the specified layer (which is of shape: [dim])
//     pub(in crate::kvcache_new) fn with_v_on_layer<R, F: FnOnce(&Tensor<P>) -> R>(&self, layer_idx: usize, op: F) -> R {
//         let v = self.v_on_layer(layer_idx);
//         op(&v)
//     }
// }

/// Unit number type that indicates how many [`KvCacheUnit`]s can be in a [`KvCacheBlock`].
pub type BlockSize = usize;

#[derive(Getters, CopyGetters)]
pub struct KvCacheBlock<TID: Copy, P: Num + Default + Copy, const BLOCK_SIZE: BlockSize = 4> {
    #[getset(get = "pub")]
    token_ids: [TID; BLOCK_SIZE],
    // units: [KvCacheUnit<P>; BLOCK_SIZE],
    #[getset(get = "pub")]
    k: Tensor<P>, // [BLOCK_SIZE, layers, n_kv_head * dqkv]
    #[getset(get = "pub")]
    v: Tensor<P>, // [BLOCK_SIZE, layers, n_kv_head * dqkv]
    #[getset(get_copy = "pub")]
    used_token_num: BlockSize,
}
impl<T: Copy + Default, P: Num + Default + Copy, const BLOCK_SIZE: BlockSize>
    KvCacheBlock<T, P, BLOCK_SIZE>
{
    pub fn default(layer_num: usize, dim: usize) -> Self {
        KvCacheBlock {
            token_ids: [T::default(); BLOCK_SIZE],
            k: Tensor::default(&[BLOCK_SIZE, layer_num, dim]),
            v: Tensor::default(&[BLOCK_SIZE, layer_num, dim]),
            // units: core::array::from_fn(|_| KvCacheUnit::default(layer_num, dim)),
            used_token_num: 0,
        }
    }
}
impl<T: Copy, P: Num + Default + Copy, const BLOCK_SIZE: BlockSize> KvCacheBlock<T, P, BLOCK_SIZE> {
    fn check_copy_shape(&self, other: &KvCacheBlock<T, P, BLOCK_SIZE>) {
        debug_assert_eq!(self.k.shape(), other.k.shape());
        debug_assert_eq!(self.v.shape(), other.v.shape());
    }

    pub fn append_from(&mut self, other: &KvCacheBlock<T, P, BLOCK_SIZE>, range: Range<usize>) {
        let range = range.start..range.end.min(other.used_token_num as usize);
        self.check_copy_shape(other);

        let len = range.len().min(self.left_unit_num());
        let range = range.start..range.start + len;

        let dst_offset = self.k.to_offset(&[self.used_token_num, 0, 0]);
        let src_offset = other.k.to_offset(&[range.start, 0, 0]);
        let buf_shape = [len, self.k.shape()[1], self.k.shape()[2]];
        {
            // copy K
            let mut dst = self.k.slice(dst_offset, &buf_shape);
            let src = other.k.slice(src_offset, &buf_shape);
            unsafe {
                dst.data_mut().copy_from_slice(src.data());
            }
        }
        {
            // copy V
            let mut dst = self.v.slice(dst_offset, &buf_shape);
            let src = other.v.slice(src_offset, &buf_shape);
            unsafe {
                dst.data_mut().copy_from_slice(src.data());
            }
        }
        self.token_ids[self.used_token_num as usize..][..len]
            .copy_from_slice(&other.token_ids[range]);
        self.used_token_num += len as BlockSize;
    }

    pub fn copy_from(&mut self, other: &KvCacheBlock<T, P, BLOCK_SIZE>, range: Range<usize>) {
        self.check_copy_shape(other);
        let range = range.start..range.end.min(other.used_token_num as usize);
        let len = range.end - range.start;

        let src_offset = other.k.to_offset(&[range.start, 0, 0]);
        let buf_shape = [len, self.k.shape()[1], self.k.shape()[2]];
        {
            // copy K
            let mut dst = self.k.slice(0, &buf_shape);
            let src = other.k.slice(src_offset, &buf_shape);
            unsafe {
                dst.data_mut().copy_from_slice(src.data());
            }
        }
        {
            // copy V
            let mut dst = self.v.slice(0, &buf_shape);
            let src = other.v.slice(src_offset, &buf_shape);
            unsafe {
                dst.data_mut().copy_from_slice(src.data());
            }
        }
        self.token_ids[..len].copy_from_slice(&other.token_ids[range]);
        self.used_token_num = len as BlockSize;
    }

    pub fn reserve_for(&mut self, token_id: &T) -> Result<(), &'static str> {
        if self.is_full() {
            return Err("block is full");
        }
        self.token_ids[self.used_token_num as usize] = *token_id;
        self.used_token_num += 1;
        Ok(())
    }
    pub fn reserve_for_slice(&mut self, token_ids: &[T]) -> Result<(), &'static str> {
        let token_len = token_ids.len();
        if token_len > self.left_unit_num() as usize {
            return Err("Left space is not enough for the given token_ids");
        }

        self.token_ids[self.used_token_num as usize..][..token_len].copy_from_slice(token_ids);
        self.used_token_num += token_len as BlockSize;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        self.used_token_num -= 1;
        Some(self.token_ids[self.used_token_num as usize])
    }
    pub fn clear(&mut self) {
        self.used_token_num = 0;
    }

    pub const fn left_unit_num(&self) -> BlockSize {
        BLOCK_SIZE - self.used_token_num
    }
    pub const fn is_empty(&self) -> bool {
        self.used_token_num == 0
    }
    pub const fn is_full(&self) -> bool {
        self.used_token_num == BLOCK_SIZE
    }
}

/// A **stable** or **unstable** block of the key-value cache.
///
/// *Stable* here means that values in a tensor-like structure will not be changed anymore.
/// A [`GuardedKvCacheBlock`] can be turned into a stable one if and only if:
///
/// - it is full(i.e, [`KvCacheBlock::is_full()`] returns true);
/// - all the k/v cache of the tokens in the block are *stable*.
///
/// More specifically, if the k/v cache of a token has been calculated through a round of "attention" layers,
/// then the k/v cache of the token can be considered to be *stable*.
/// If all the k/v cache of the tokens in the block are *stable*,
/// then the block can be turned into a stable one.
pub enum GuardedKvCacheBlock<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize> {
    Stable(KvCacheBlock<T, P, BLOCK_SIZE>),
    Unstable(UnstableKvCacheBlock<T, P, BLOCK_SIZE>),
}
#[derive(Getters, CopyGetters, MutGetters)]
pub struct UnstableKvCacheBlock<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>
{
    #[getset(get = "pub", get_mut = "pub")]
    block: KvCacheBlock<T, P, BLOCK_SIZE>,
    pub stablized_token_num: BlockSize,
}
impl<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>
    UnstableKvCacheBlock<T, P, BLOCK_SIZE>
{
    pub fn new(block: KvCacheBlock<T, P, BLOCK_SIZE>) -> Self {
        UnstableKvCacheBlock {
            block,
            stablized_token_num: 0,
        }
    }
    pub fn can_be_stable(&self) -> bool {
        self.block.is_full() && self.stablized_token_num == self.block.used_token_num
    }
    pub fn into_stable(self) -> Result<KvCacheBlock<T, P, BLOCK_SIZE>, &'static str> {
        if self.block.is_full() && self.can_be_stable() {
            Ok(self.block)
        } else {
            Err("The block can be turned into a stable one only if all the k/v cache of the tokens in the block are stable.")
        }
    }
}
impl<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>
    GuardedKvCacheBlock<T, P, BLOCK_SIZE>
{
    pub fn new_stable(block: KvCacheBlock<T, P, BLOCK_SIZE>) -> Self {
        GuardedKvCacheBlock::Stable(block)
    }
    pub fn new_unstable(block: KvCacheBlock<T, P, BLOCK_SIZE>) -> Self {
        GuardedKvCacheBlock::Unstable(UnstableKvCacheBlock {
            block,
            stablized_token_num: 0,
        })
    }

    pub fn into_stable(self) -> Result<Self, &'static str> {
        match self {
            GuardedKvCacheBlock::Unstable(ublock) if ublock.can_be_stable() => {
                Ok(GuardedKvCacheBlock::Stable(ublock.block))
            }
            GuardedKvCacheBlock::Unstable(_) => {
                Err("Blocks can be turned into stable ones only if they are full and all the kvcache of the tokens in the block are stable.")
            }
            GuardedKvCacheBlock::Stable(_) => Ok(self),
        }
    }
    pub fn unwrap_stable(self) -> KvCacheBlock<T, P, BLOCK_SIZE> {
        match self {
            GuardedKvCacheBlock::Stable(block) => block,
            GuardedKvCacheBlock::Unstable(_) => Self::unwrap_stable(self.into_stable().unwrap()),
        }
    }

    pub const fn is_stable(&self) -> bool {
        matches!(self, GuardedKvCacheBlock::Stable(_))
    }
    pub const fn is_unstable(&self) -> bool {
        matches!(self, GuardedKvCacheBlock::Unstable(_))
    }

    pub fn read_kv_cache_block<R>(
        &self,
        op: impl FnOnce(&KvCacheBlock<T, P, BLOCK_SIZE>, BlockSize) -> R,
    ) -> R {
        match self {
            GuardedKvCacheBlock::Stable(block) => op(block, BLOCK_SIZE),
            GuardedKvCacheBlock::Unstable(UnstableKvCacheBlock {
                stablized_token_num,
                ref block,
            }) => op(block, *stablized_token_num),
        }
    }

    /// Reads the *used* token_ids of the block.
    pub fn read_token_ids<R>(&self, op: impl FnOnce(&[T], BlockSize) -> R) -> R {
        self.read_kv_cache_block(|block, stablized_token_num| {
            debug_assert!(stablized_token_num <= block.used_token_num() as usize);
            op(
                &block.token_ids[..block.used_token_num as usize],
                stablized_token_num,
            )
        })
    }

    // /// Operates on the *used* units of the block.
    // pub fn with_kv_cache_units<R>(&self, op: impl FnOnce(&[KvCacheUnit<P>]) -> R) -> R {
    //     self.with_kv_cache_block(|block| op(&block.units[..block.used_unit_num]))
    // }

    /// Reads the *used* k cache of the block.
    pub fn read_k_cache<R>(&self, op: impl FnOnce(&Tensor<P>, BlockSize) -> R) -> R {
        self.read_kv_cache_block(|block, stablized_token_num| {
            let layer_num = block.k.shape()[1];
            let dim = block.k.shape()[2];
            op(
                &block.k.slice(0, &[block.used_token_num, layer_num, dim]),
                stablized_token_num,
            )
        })
    }
    /// Reads the *used* v cache of the block.
    pub fn read_v_cache<R>(&self, op: impl FnOnce(&Tensor<P>, BlockSize) -> R) -> R {
        self.read_kv_cache_block(|block, stablized_token_num| {
            let layer_num = block.v.shape()[1];
            let dim = block.v.shape()[2];
            op(
                &block.v.slice(0, &[block.used_token_num, layer_num, dim]),
                stablized_token_num,
            )
        })
    }

    /// Writes on the unstable block.
    pub fn write_kv_cache_block<R>(
        &mut self,
        op: impl FnOnce(&mut KvCacheBlock<T, P, BLOCK_SIZE>, &mut BlockSize) -> R,
    ) -> Result<R, &'static str> {
        match self {
            GuardedKvCacheBlock::Stable(_) => Err("Should not mutate a stable block"),
            GuardedKvCacheBlock::Unstable(UnstableKvCacheBlock {
                stablized_token_num,
                block,
            }) => Ok(op(block, stablized_token_num)),
        }
    }
}
