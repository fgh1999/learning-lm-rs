use crate::{
    config::LlamaConfigJson,
    kvcache::{BlockSize, LocalKvCache},
    operators::{self as OP, cartesian_product2},
    params::LlamaParams,
    tensor::{Tensor, TensorIndex, TensorView, WritableTensorView},
};
use getset::{CopyGetters, Getters};
use num_traits::{Float, Num};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::{
    fs::File,
    ops::{AddAssign, DivAssign, MulAssign},
    path::Path,
    sync::Mutex,
};

#[derive(Getters)]
pub struct Llama<T: Num> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LlamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
    #[getset(get = "pub")]
    perf_info: Mutex<PerfInfo>,
}

#[derive(Getters, Default, Debug)]
pub struct PerfInfo {
    #[getset(get = "pub")]
    total_generation_duration: Option<std::time::Duration>,
    #[getset(get = "pub")]
    prompt_duration: Option<std::time::Duration>,
}

pub trait LmModel<TID: Num + Copy + Eq, P: Float + Default + Copy> {
    /// Returns the logits of the next token
    /// if the given last cache block in the cache is the last block of the sequence.
    fn forward_on_last_cache_block<const BLOCK_SIZE: BlockSize>(
        &self,
        cache: &mut LocalKvCache<TID, P, BLOCK_SIZE>,
        buffer: &mut ForwardBuffer<P>,
    ) -> Option<Tensor<P>>;
    fn alloc_forward_buffer(&self, seq_len: usize) -> ForwardBuffer<P>;
    fn layer_num(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn kv_dim(&self) -> usize;
    fn bos_token_id(&self) -> TID;
    fn eos_token_id(&self) -> TID;
}

#[derive(CopyGetters)]
pub struct ForwardBuffer<P: Float + Default + Copy> {
    pub residual: Tensor<P>,
    pub hidden_states: Tensor<P>,
    pub q_buf: Tensor<P>,
    pub att_scores: Tensor<P>,
    pub gate_buf: Tensor<P>,
    pub up_buf: Tensor<P>,
    /// The maximum token sequence length that this buffer can handle
    #[getset(get_copy = "pub")]
    max_seq_len: usize,
}

impl<
        P: 'static
            + Float
            + std::iter::Sum
            + Sync
            + Send
            + MulAssign
            + DivAssign
            + AddAssign
            + Copy
            + Clone
            + Default,
    > LmModel<u32, P> for Llama<P>
{
    fn alloc_forward_buffer(&self, seq_len: usize) -> ForwardBuffer<P> {
        let n_groups = self.n_q_h / self.n_kv_h;
        ForwardBuffer {
            residual: Tensor::<P>::default(&[seq_len, self.d]),
            hidden_states: Tensor::<P>::default(&[seq_len, self.d]),
            q_buf: Tensor::<P>::default(&[seq_len, self.n_q_h * self.dqkv]),
            att_scores: Tensor::<P>::default(&[self.n_kv_h, n_groups, seq_len, self.max_seq_len]),
            gate_buf: Tensor::<P>::default(&[seq_len, self.di]),
            up_buf: Tensor::<P>::default(&[seq_len, self.di]),
            max_seq_len: seq_len,
        }
    }

    fn forward_on_last_cache_block<const BLOCK_SIZE: BlockSize>(
        &self,
        cache: &mut LocalKvCache<u32, P, BLOCK_SIZE>,
        buffer: &mut ForwardBuffer<P>,
    ) -> Option<Tensor<P>> {
        let appended_input =
            cache
                .last_block
                .as_ref()
                .unwrap()
                .read_token_ids(|token_ids, stablized_num| {
                    let new_token_ids = token_ids[stablized_num..].to_vec();
                    let l = new_token_ids.len();
                    Tensor::<u32>::new(new_token_ids, &[1, l])
                });
        let seq_len = appended_input.size();
        debug_assert!(seq_len <= BLOCK_SIZE);
        let past_seq_len = cache.total_stable_common_token_num();
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Picks pre-allocated buffers
        debug_assert!(buffer.max_seq_len >= seq_len);
        debug_assert!(buffer
            .att_scores
            .shape()
            .iter()
            .zip([self.n_kv_h, n_groups, seq_len, total_seq_len].iter())
            .all(|(b, r)| b >= r));
        let mut residual = buffer.residual.slice(0, &[seq_len, self.d]);
        let mut hidden_states = buffer.hidden_states.slice(0, &[seq_len, self.d]);
        let mut q_buf = buffer.q_buf.slice(0, &[seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores = buffer
            .att_scores
            .slice(0, &[self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = buffer.gate_buf.slice(0, &[seq_len, self.di]);
        let mut up_buf = buffer.up_buf.slice(0, &[seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, &appended_input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            // Multi-head self-attention
            {
                OP::rms_norm(
                    &mut hidden_states,
                    &residual,
                    &self.params.rms_att_w[layer],
                    self.eps,
                );

                let q = q_buf.reshape(&[seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
                OP::matmul_transb(
                    q,
                    P::zero(),
                    &hidden_states,
                    &self.params.wq[layer],
                    P::one(),
                );
                OP::rope(
                    q.reshape(&[seq_len, self.n_q_h, self.dqkv]),
                    past_seq_len,
                    self.rope_theta,
                );

                cache
                    .last_block
                    .as_mut()
                    .unwrap()
                    .write_kv_cache_block(|block, &mut stablized_num| {
                        let c = block.k();
                        let c = c.slice(
                            c.to_offset(&[stablized_num, 0, 0]),
                            &[seq_len, self.layer_num(), self.kv_dim()],
                        );
                        // [seq_len, layer_num, kv_dim] ---layer--> [seq_len, kv_dim]
                        let c = (0..seq_len)
                            .map(|i| c.slice(c.to_offset(&[i, layer, 0]), &[self.kv_dim()]))
                            .collect::<Vec<_>>();
                        let mut c = VecTensor::new(c);

                        OP::matmul_transb(
                            &mut c,
                            P::zero(),
                            &hidden_states,
                            &self.params.wk[layer],
                            P::one(),
                        );
                        let mut c = c.slice(0, &[seq_len, self.n_kv_h, self.dqkv]);
                        OP::rope(&mut c, past_seq_len, self.rope_theta);
                    })
                    .unwrap();
                cache
                    .last_block
                    .as_mut()
                    .unwrap()
                    .write_kv_cache_block(|block, &mut stablized_num| {
                        let c = block.v();
                        let c = c.slice(
                            c.to_offset(&[stablized_num, 0, 0]),
                            &[seq_len, self.layer_num(), self.kv_dim()],
                        );
                        // [seq_len, layer_num, kv_dim] ---layer--> [seq_len, kv_dim]
                        let c = (0..seq_len)
                            .map(|i| c.slice(c.to_offset(&[i, layer, 0]), &[self.kv_dim()]))
                            .collect::<Vec<_>>();
                        let mut c = VecTensor::new(c);

                        OP::matmul_transb(
                            &mut c,
                            P::zero(),
                            &hidden_states,
                            &self.params.wv[layer],
                            P::one(),
                        );
                    })
                    .unwrap();

                // all used common units in the local cache
                let full_k = cache
                    .last_block
                    .as_ref()
                    .unwrap()
                    .read_kv_cache_block(|block, _| {
                        let stable_blocks = cache.stable_blocks();
                        let stable_c_vec =
                            stable_blocks
                                .iter()
                                .flat_map(|&(stable_block, common_len)| {
                                    let c = stable_block.k();
                                    (0..common_len).map(|i| {
                                        c.slice(c.to_offset(&[i, layer, 0]), &[self.kv_dim()])
                                    })
                                });
                        let mut k_vec = Vec::from_iter(stable_c_vec);
                        k_vec.extend({
                            let c = block.k();
                            (0..block.used_token_num())
                                .map(|i| c.slice(c.to_offset(&[i, layer, 0]), &[self.kv_dim()]))
                        });
                        VecTensor::new(k_vec)
                    }); // (total_seq, n_kv_h * dqkv)
                let full_v = cache
                    .last_block
                    .as_ref()
                    .unwrap()
                    .read_kv_cache_block(|block, _| {
                        let stable_blocks = cache.stable_blocks();
                        let stable_c_vec =
                            stable_blocks
                                .iter()
                                .flat_map(|&(stable_block, common_len)| {
                                    let c = stable_block.v();
                                    (0..common_len).map(|i| {
                                        c.slice(c.to_offset(&[i, layer, 0]), &[self.kv_dim()])
                                    })
                                });
                        let mut v_vec = Vec::from_iter(stable_c_vec);
                        v_vec.extend({
                            let c = block.v();
                            (0..block.used_token_num())
                                .map(|i| c.slice(c.to_offset(&[i, layer, 0]), &[self.kv_dim()]))
                        });
                        VecTensor::new(v_vec)
                    }); // (total_seq, n_kv_h * dqkv)

                self_attention(
                    &mut hidden_states,
                    &mut att_scores,
                    q,
                    &full_k,
                    &full_v,
                    self.n_kv_h,
                    n_groups,
                    seq_len,
                    total_seq_len,
                    self.dqkv,
                );
                OP::matmul_transb(
                    &mut residual,
                    P::one(),
                    &hidden_states,
                    &self.params.wo[layer],
                    P::one(),
                );
            }

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        cache
            .last_block
            .as_mut()
            .unwrap()
            .write_kv_cache_block(|block, stablized_num| {
                debug_assert_eq!(block.used_token_num(), *stablized_num + seq_len);
                *stablized_num += seq_len;
            })
            .unwrap();

        if !cache.remaining_token_ids().is_empty() {
            // The last block is not at the last of the sequence.
            // Still prefilling, not ready to generate the next token.
            return None;
        }

        let mut hidden_states =
            hidden_states.slice(hidden_states.to_offset(&[seq_len - 1, 0]), &[1, self.d]);
        let residual = residual.slice(residual.to_offset(&[seq_len - 1, 0]), &[1, self.d]);
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<P>::default(&[1, self.vocab]);
        OP::matmul_transb(
            &mut logits,
            P::zero(),
            &hidden_states,
            &self.params.lm_head,
            P::one(),
        );
        Some(logits)
    }

    fn layer_num(&self) -> usize {
        self.n_layers
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn kv_dim(&self) -> usize {
        self.n_kv_h * self.dqkv
    }
    fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }
    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

macro_rules! impl_from_safetensors_for_Llama {
    ($Param:ty) => {
        impl Llama<$Param> {
            pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
                let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
                let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
                let model_file =
                    std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
                let safetensor = SafeTensors::deserialize(&model_file).unwrap();

                assert!(config.num_attention_heads % config.num_key_value_heads == 0);
                Self {
                    vocab: config.vocab_size,
                    n_layers: config.num_hidden_layers,
                    n_q_h: config.num_attention_heads,
                    n_kv_h: config.num_key_value_heads,
                    d: config.hidden_size,
                    dqkv: config.hidden_size / config.num_attention_heads,
                    di: config.intermediate_size,
                    eps: config.rms_norm_eps,
                    rope_theta: config.rope_theta,
                    max_seq_len: config.max_position_embeddings,
                    params: LlamaParams::<$Param>::from_safetensors(&safetensor, &config),
                    bos_token_id: config.bos_token_id,
                    eos_token_id: config.eos_token_id,
                    perf_info: Mutex::new(PerfInfo::default()),
                }
            }
        }
    };
}
impl_from_safetensors_for_Llama!(f32);
impl_from_safetensors_for_Llama!(f64);
impl_from_safetensors_for_Llama!(half::f16);
impl_from_safetensors_for_Llama!(half::bf16);

#[allow(clippy::too_many_arguments)]
fn self_attention<
    P: Float + std::iter::Sum + Sync + Send + MulAssign + DivAssign + AddAssign,
    T0: TensorView<P> + Sync,
    T1: TensorView<P> + Sync,
>(
    hidden_states: &mut Tensor<P>, // (seq, n_kv_h * n_groups * dqkv) as return value
    att_scores: &mut Tensor<P>,    // (n_kv_h, n_groups, seq, total_seq) as buffer
    q: &T0,                        // (seq, n_kv_h * n_groups, dqkv)
    k: &T1,                        // (total_seq, n_kv_h * dqkv)
    v: &T1,                        // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    assert!(k.shape()[0] >= total_seq_len);
    assert!(v.shape()[0] >= total_seq_len);

    assert!(q.shape()[0] >= seq_len);
    assert!(q.shape()[1] == n_kv_h * n_groups && q.shape()[2] == dqkv);
    let q = q.slice(0, &[seq_len, n_kv_h, n_groups, dqkv]);
    assert!(k.shape()[1] == n_kv_h * dqkv);
    let k = k.slice(0, &[total_seq_len, n_kv_h, dqkv]);
    let dqkv_root = P::from(dqkv).unwrap().sqrt();

    let att_indices: Vec<_> = cartesian_product2(0..seq_len, 0..total_seq_len)
        // skip masked calculation
        .filter(|&(seq_idx, tseq_idx)| (total_seq_len - seq_len) + seq_idx >= tseq_idx)
        .collect();

    cartesian_product2(0..n_kv_h, 0..n_groups).for_each(|(kv_idx, g_idx)| {
        #[cfg(feature = "rayon")]
        let att_idx_iter = att_indices.par_iter();
        #[cfg(not(feature = "rayon"))]
        let att_idx_iter = att_indices.iter();

        att_idx_iter.for_each(|&(seq_idx, tseq_idx)| {
            let q_vec = q.slice(q.to_offset(&[seq_idx, kv_idx, g_idx, 0]), &[dqkv]);
            let k_vec = k.slice(k.to_offset(&[tseq_idx, kv_idx, 0]), &[dqkv]);
            let att_score = OP::dot(&q_vec, &k_vec) / dqkv_root;
            let mut a = att_scores.slice(
                att_scores.to_offset(&[kv_idx, g_idx, seq_idx, tseq_idx]),
                &[1],
            );
            unsafe {
                a.data_mut()[0] = att_score;
            }
        });
    });
    OP::masked_softmax(att_scores);

    let v = v.slice(0, &[total_seq_len, n_kv_h, dqkv]);
    unsafe {
        hidden_states.erase();
    }
    let hidden = hidden_states.slice(0, &[seq_len, n_kv_h, n_groups, dqkv]);

    let g_seq_indices = cartesian_product2(0..n_groups, 0..seq_len);
    #[cfg(feature = "rayon")]
    let g_seq_indices = g_seq_indices.collect::<Vec<_>>().into_par_iter();
    g_seq_indices.for_each(|(g_idx, seq_idx)| {
        // no parallelization on `tseq_idx` because of the data race on `hidden_vec`
        cartesian_product2(0..n_kv_h, 0..total_seq_len).for_each(|(kv_idx, tseq_idx)| {
            let v_vec = v.slice(v.to_offset(&[tseq_idx, kv_idx, 0]), &[dqkv]);
            let att_score = att_scores[&[kv_idx, g_idx, seq_idx, tseq_idx]];
            let mut hidden_vec =
                hidden.slice(hidden.to_offset(&[seq_idx, kv_idx, g_idx, 0]), &[dqkv]);
            unsafe {
                hidden_vec
                    .data_mut()
                    .iter_mut()
                    .zip(v_vec.data_iter())
                    .for_each(|(h, &val)| *h += att_score.mul(val))
            };
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn mlp<P: Float + std::iter::Sum + Sync + Send + MulAssign>(
    residual: &mut Tensor<P>,      // as input and output
    hidden_states: &mut Tensor<P>, // as buffer
    gate: &mut Tensor<P>,
    up: &mut Tensor<P>,
    w_up: &Tensor<P>,
    w_down: &Tensor<P>,
    w_gate: &Tensor<P>,
    rms_w: &Tensor<P>,
    eps: impl Float,
) {
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    OP::matmul_transb(gate, P::zero(), hidden_states, w_gate, P::one());
    OP::matmul_transb(up, P::zero(), hidden_states, w_up, P::one());
    OP::silu(up, gate);
    OP::matmul_transb(residual, P::one(), up, w_down, P::one());
}

struct VecTensor<P: Num> {
    data: Vec<Tensor<P>>,
    shape: Vec<usize>,
    storage_offset: usize,
}
impl<P: Num> VecTensor<P> {
    pub fn new(data: Vec<Tensor<P>>) -> Self {
        Self {
            shape: {
                debug_assert!(!data.is_empty());
                let mut shape = vec![data.len()];
                shape.extend_from_slice(data[0].shape());
                shape
            },
            data,
            storage_offset: 0,
        }
    }
}
impl<P: Num> TensorIndex for VecTensor<P> {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}
impl<P: Num + Clone> TensorView<P> for VecTensor<P> {
    fn data_at(&self, idx: &[usize]) -> &P {
        let offset = self.storage_offset + self.to_offset(idx);
        let size_per_vec = self.data[0].size();
        let v = &self.data[offset / size_per_vec];
        let idx_within_vec = Self::offset_to_index(offset % size_per_vec, v.shape());
        v.data_at(&idx_within_vec)
    }
    fn data_iter<'a>(&'a self) -> impl Iterator<Item = &'a P>
    where
        P: 'a,
    {
        self.data
            .iter()
            .flat_map(|t| t.data_iter())
            .skip(self.storage_offset)
            .take(self.size())
    }
    fn slice(&self, start: usize, shape: &[usize]) -> Self {
        let length: usize = shape.iter().product();
        assert!(length > 0 && length <= self.size() && start <= self.size() - length);
        let offset = self.storage_offset + start;
        let end = offset + length;
        let size_per_vec = self.data[0].size();

        let data = self.data[offset / size_per_vec..((end - 1) / size_per_vec + 1)].to_vec();
        Self {
            data,
            shape: shape.to_vec(),
            storage_offset: offset % size_per_vec,
        }
    }
}
unsafe impl<P: Num + Copy> WritableTensorView<P> for VecTensor<P> {
    unsafe fn with_data_mut_at(&mut self, idx: &[usize], op: impl FnOnce(&P) -> P) -> P {
        let data_ptr = self.data_at(idx);
        let prev_val = *data_ptr;
        let data_ptr = data_ptr as *const P as *mut P;
        data_ptr.write(op(&prev_val));
        prev_val
    }
    unsafe fn data_iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut P>
    where
        P: 'a,
    {
        self.data_iter()
            .map(|p| (p as *const P as *mut P).as_mut().unwrap())
    }
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-6
    ))
}

#[test]
pub fn test_load_safetensors_from_story_model() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("../models").join("story");
    println!("{:?}", model_dir);
    let model = Llama::<f32>::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}

#[test]
pub fn test_load_safetensors_from_chat_model() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("../models").join("chat");
    if !model_dir.exists() {
        use std::io::Write;
        std::io::stderr()
            .write(b"Please download the chat model first.\n")
            .unwrap();
        return;
    }

    let model = Llama::<f32>::from_safetensors(model_dir);
    assert_eq!(model.vocab, 32002);
    assert_eq!(model.n_layers, 10);
    assert_eq!(model.n_q_h, 12);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 312);
    assert_eq!(model.dqkv, 26);
    assert_eq!(model.di, 1092);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &-0.018187439,
        1e-6
    ));
    assert!(float_eq(
        &model.params.embedding_table[&[31012, 55]],
        &-0.009104947,
        1e-6
    ));
    assert!(float_eq(
        &model.params.lm_head[&[20100, 3]],
        &-0.032863498,
        1e-6
    ));
}
