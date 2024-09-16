use super::{
    block::{BlockSize, GuardedKvCacheBlock, KvCacheBlock},
    try_add_root_kv_cache,
};
use getset::{CopyGetters, Getters};
use num_traits::Num;
use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

// Lock Order: always visit `.block` before `.children`
#[derive(Getters, CopyGetters)]
pub struct RadixAttentionTreeNode<
    T: Copy + Eq,
    P: Num + Copy + Default,
    const BLOCK_SIZE: BlockSize,
> {
    #[getset(get_copy = "pub")]
    common_prefix_len_with_parent: usize,
    #[getset(get = "pub")]
    block: KvCacheBlock<T, P, BLOCK_SIZE>,
    children: RwLock<VecDeque<Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>>>,
}

impl<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize> PartialEq
    for RadixAttentionTreeNode<T, P, BLOCK_SIZE>
{
    fn eq(&self, other: &Self) -> bool {
        self.common_prefix_len_with_parent == other.common_prefix_len_with_parent
            && self.block.token_ids() == other.block.token_ids()
    }
}

impl<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>
    RadixAttentionTreeNode<T, P, BLOCK_SIZE>
{
    pub fn new(
        common_prefix_len_with_parent: usize,
        block: KvCacheBlock<T, P, BLOCK_SIZE>,
    ) -> Self {
        RadixAttentionTreeNode {
            common_prefix_len_with_parent,
            block,
            children: RwLock::new(VecDeque::new()),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.read().unwrap().is_empty()
    }

    // pub fn token_id_seq(&self) -> impl Iterator<Item = T> + '_ {
    //     self.block.with_kv_cache_block(|block| block.units().iter().map(|unit| unit.token_id()))
    // }

    pub const fn max_seq_len(&self) -> usize {
        BLOCK_SIZE
    }

    /// Returns the largest common prefix length of *the node* and the given index sequence.
    fn largest_common_prefix_length_with<'a>(&self, other: impl Iterator<Item = &'a T>) -> usize
    where
        T: 'a,
    {
        let token_ids = self.block.token_ids();
        match token_ids.iter().zip(other).position(|(a, b)| !a.eq(&b)) {
            Some(i) => i,
            None => token_ids.len(),
        }
    }

    // move stable child into self
    fn add_child(
        &self,
        common_prefix_len: usize,
        block: KvCacheBlock<T, P, BLOCK_SIZE>,
    ) -> Result<
        Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>, // successfully merged child node
        (
            Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>, // priorly merged child node with common prefix
            KvCacheBlock<T, P, BLOCK_SIZE>,                // the block passed in to merge
        ),
    > {
        let children = &mut self.children.write().unwrap();
        match children
            .iter()
            .find(|child| child.largest_common_prefix_length_with(block.token_ids().iter()) > 0)
        {
            Some(conflict_child) => Err((conflict_child.clone(), block)),
            None => {
                let new_child = RadixAttentionTreeNode::new(common_prefix_len, block);
                let new_child = Arc::new(new_child);
                children.push_front(new_child.clone());
                Ok(new_child)
            }
        }
    }
}

pub struct RadixAttentionForest<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>
{
    roots: RwLock<VecDeque<Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>>>,
}
impl<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize> Default
    for RadixAttentionForest<T, P, BLOCK_SIZE>
{
    fn default() -> Self {
        RadixAttentionForest {
            roots: RwLock::new(VecDeque::new()),
        }
    }
}
impl<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize>
    RadixAttentionForest<T, P, BLOCK_SIZE>
{
    pub fn try_add_root(
        &self,
        block: KvCacheBlock<T, P, BLOCK_SIZE>,
    ) -> Result<
        Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>, // successfully merged child node
        (
            Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>, // priorly merged child node with common prefix
            KvCacheBlock<T, P, BLOCK_SIZE>,                // the block passed in to merge
        ),
    > {
        let roots = &mut self.roots.write().unwrap();
        match roots
            .iter()
            .find(|root| root.largest_common_prefix_length_with(block.token_ids().iter()) > 0)
        {
            Some(conflict_root) => Err((conflict_root.clone(), block)),
            None => {
                let new_root = Arc::new(RadixAttentionTreeNode::new(0, block));
                roots.push_front(new_root.clone());
                Ok(new_root)
            }
        }
    }
    pub fn get_cache_for(
        &self,
        token_ids: &[T],
        layer_num: usize,
        kv_dim: usize,
    ) -> LocalKvCache<T, P, BLOCK_SIZE> {
        let mut stable_nodes = Vec::new();
        let mut remaining_token_ids = token_ids;

        let find_subtree_with_common_prefix =
            |trees: &VecDeque<Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>>, token_ids: &[T]| {
                trees
                    .iter()
                    .find(|tree| tree.largest_common_prefix_length_with(token_ids.iter()) > 0)
                    .cloned()
            };

        let mut tree_with_common_prefix =
            find_subtree_with_common_prefix(&self.roots.read().unwrap(), remaining_token_ids);
        while !remaining_token_ids.is_empty() && tree_with_common_prefix.is_some() {
            let node = tree_with_common_prefix.unwrap();
            let common_len = node.largest_common_prefix_length_with(remaining_token_ids.iter());
            debug_assert!(common_len <= BLOCK_SIZE);

            remaining_token_ids = &remaining_token_ids[common_len..];
            tree_with_common_prefix = find_subtree_with_common_prefix(
                &node.children.read().unwrap(),
                remaining_token_ids,
            );

            stable_nodes.push(StableKvCacheNode { node, common_len });
        }

        LocalKvCache {
            stable_nodes,
            last_block: None,
            remaining_token_ids: remaining_token_ids.to_vec().into(),
            layer_num,
            kv_dim,
        }
    }
}

#[derive(Getters)]
pub struct LocalKvCache<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize> {
    stable_nodes: Vec<StableKvCacheNode<T, P, BLOCK_SIZE>>,
    /// Unstable last block
    pub last_block: Option<GuardedKvCacheBlock<T, P, BLOCK_SIZE>>,
    #[getset(get = "pub")]
    remaining_token_ids: VecDeque<T>,

    layer_num: usize,
    kv_dim: usize,
}

struct StableKvCacheNode<T: Copy + Eq, P: Num + Copy + Default, const BLOCK_SIZE: BlockSize> {
    node: Arc<RadixAttentionTreeNode<T, P, BLOCK_SIZE>>,
    /// The length of token_ids that this node is used in the local cache
    common_len: usize,
}
impl<
        T: 'static + Copy + Eq + Default,
        P: 'static + Num + Copy + Default,
        const BLOCK_SIZE: BlockSize,
    > LocalKvCache<T, P, BLOCK_SIZE>
{
    pub fn stable_blocks(&self) -> Vec<(&KvCacheBlock<T, P, BLOCK_SIZE>, usize)> {
        self.stable_nodes
            .iter()
            .map(|node| (node.node.block(), node.common_len))
            .collect()
    }
    pub fn total_stable_common_token_num(&self) -> usize {
        self.stable_nodes
            .iter()
            .map(|node| node.common_len)
            .sum::<usize>()
            + self.last_block.as_ref().map_or(0, |block| match block {
                GuardedKvCacheBlock::Stable(block) => block.used_token_num(),
                GuardedKvCacheBlock::Unstable(block) => block.stablized_token_num,
            })
    }

    pub fn append_token_ids(&mut self, token_ids: &[T]) {
        self.remaining_token_ids.extend(token_ids);
    }

    fn can_sync_last_block_into_global(&self) -> bool {
        self.last_block.as_ref().map_or(false, |block| match block {
            GuardedKvCacheBlock::Stable(_) => true,
            GuardedKvCacheBlock::Unstable(ublock) => ublock.can_be_stable(),
        })
    }

    fn new_last_block(&self, token_ids: &[T]) -> GuardedKvCacheBlock<T, P, BLOCK_SIZE> {
        debug_assert!(token_ids.len() <= BLOCK_SIZE);
        let mut block = KvCacheBlock::default(self.layer_num, self.kv_dim);
        block.reserve_for_slice(token_ids).unwrap();
        GuardedKvCacheBlock::new_unstable(block)
    }

    /// Usually used after a sync with global kvcache
    /// Loads the last block from the remaining token ids,
    /// and returns whether the last block is loaded with new token ids.
    pub fn load_last_block(&mut self) -> bool {
        if self.last_block.is_none() {
            let token_ids = self
                .remaining_token_ids
                .iter()
                .take(BLOCK_SIZE)
                .copied()
                .collect::<Vec<_>>();
            token_ids.iter().for_each(|_| {
                self.remaining_token_ids.pop_front();
            });
            if token_ids.is_empty() {
                false
            } else {
                self.last_block = Some(self.new_last_block(&token_ids));
                true
            }
        } else {
            match self.last_block.as_mut().unwrap() {
                GuardedKvCacheBlock::Stable(_) => {
                    panic!("Should not load the last block when it is stable, i.e. waiting for sync with global");
                }
                GuardedKvCacheBlock::Unstable(ublock) => {
                    debug_assert!(!ublock.can_be_stable());
                    let token_ids = self
                        .remaining_token_ids
                        .iter()
                        .take(BLOCK_SIZE.min(ublock.block().left_unit_num()))
                        .copied()
                        .collect::<Vec<_>>();
                    token_ids.iter().for_each(|_| {
                        self.remaining_token_ids.pop_front();
                    });
                    if token_ids.is_empty() {
                        false
                    } else {
                        ublock.block_mut().reserve_for_slice(&token_ids).unwrap();
                        true
                    }
                }
            }
        }
    }

    pub fn try_sync_last_block_into_global(&mut self) {
        if !self.can_sync_last_block_into_global() {
            return;
        }

        let last_block = self.last_block.take().unwrap().unwrap_stable();
        match if let Some(parent) = self.stable_nodes.last() {
            parent.node.add_child(parent.common_len, last_block)
        } else {
            try_add_root_kv_cache(last_block).expect("Global KvCache is not initialized")
        } {
            Ok(merged_node) => {
                self.stable_nodes.push(StableKvCacheNode {
                    common_len: merged_node.max_seq_len(),
                    node: merged_node,
                });
            }
            Err((priorly_merged_node, block)) => {
                let common_len =
                    priorly_merged_node.largest_common_prefix_length_with(block.token_ids().iter());
                self.stable_nodes.push(StableKvCacheNode {
                    common_len,
                    node: priorly_merged_node,
                });

                // recycle the latter part of the failed merged block
                let mut recycled_last_block = KvCacheBlock::default(self.layer_num, self.kv_dim);
                recycled_last_block.copy_from(&block, common_len..block.used_token_num());
                self.last_block = Some(GuardedKvCacheBlock::new_unstable(recycled_last_block));
            }
        }
    }
}
