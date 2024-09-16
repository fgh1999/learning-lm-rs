mod block;
pub use block::{BlockSize, KvCacheBlock};
use std::sync::{Arc, LazyLock};
mod radix_tree;
use num_traits::Num;
pub use radix_tree::LocalKvCache;
use radix_tree::{RadixAttentionForest, RadixAttentionTreeNode};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::Mutex,
};

// type TID = u32;
pub const BLOCK_SIZE: BlockSize = 4;

struct TypeMap {
    inner: Mutex<HashMap<TypeId, Box<dyn Any>>>,
}
unsafe impl Send for TypeMap {}
unsafe impl Sync for TypeMap {}

impl TypeMap {
    fn new() -> Self {
        TypeMap {
            inner: Mutex::new(HashMap::new()),
        }
    }

    fn insert<T: 'static>(&self, value: T) {
        let map = &mut self.inner.lock().unwrap();
        map.insert(TypeId::of::<T>(), Box::new(value));
    }

    fn get_by<T: 'static, R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let map = self.inner.lock().unwrap();
        map.get(&TypeId::of::<T>())
            .map(|obj| f(obj.downcast_ref::<T>().unwrap()))
    }
}

static GLOBAL_KVCACHE: LazyLock<TypeMap> = LazyLock::new(|| {
    use half::{bf16, f16};

    let map = TypeMap::new();
    map.insert(RadixAttentionForest::<u32, f16, BLOCK_SIZE>::default());
    map.insert(RadixAttentionForest::<u32, bf16, BLOCK_SIZE>::default());
    map.insert(RadixAttentionForest::<u32, f32, BLOCK_SIZE>::default());
    map.insert(RadixAttentionForest::<u32, f64, BLOCK_SIZE>::default());
    map
});

pub fn get_local_kv_cache_for<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>(
    token_ids: &[T],
    layer_num: usize,
    kv_dim: usize,
) -> Option<LocalKvCache<T, P, BLOCK_SIZE>>
where
    RadixAttentionForest<T, P, BLOCK_SIZE>: 'static,
{
    GLOBAL_KVCACHE.get_by(|global_kvcache: &RadixAttentionForest<T, P, BLOCK_SIZE>| {
        global_kvcache.get_cache_for(token_ids, layer_num, kv_dim)
    })
}

pub fn try_add_root_kv_cache<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>(
    root_block: KvCacheBlock<T, P, BLOCK_SIZE>,
) -> Option<
    Result<
        Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>, // successfully merged child node
        (
            Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>, // priorly merged child node with common prefix
            KvCacheBlock<T, P, BLOCK_SIZE>,                // the block passed in to merge
        ),
    >,
>
where
    RadixAttentionForest<T, P, BLOCK_SIZE>: 'static,
{
    GLOBAL_KVCACHE.get_by(|global_kvcache: &RadixAttentionForest<T, P, BLOCK_SIZE>| {
        global_kvcache.try_add_root(root_block)
    })
}
