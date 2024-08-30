use crate::{
    config::LlamaConfigJson,
    kvcache::KVCache,
    operators::{self as OP, cartesian_product2},
    params::LlamaParams,
    tensor::{Tensor, TensorView},
};
use getset::Getters;
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

pub trait LmModel<TID: Num + Copy, P: Float + Default + Copy + Clone> {
    fn forward(&self, input: &Tensor<TID>, cache: &mut KVCache<TID, P>) -> Tensor<P>;
    fn layer_num(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn kv_dim(&self) -> usize;
    fn bos_token_id(&self) -> TID;
    fn eos_token_id(&self) -> TID;
}

impl<
        P: Float
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
    fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<u32, P>) -> Tensor<P> {
        let seq_len = input.size();
        let past_seq_len = cache.seq_len();
        cache.push_from(input.data()).unwrap();
        let total_seq_len = cache.seq_len();
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<P>::default(&[seq_len, self.d]);
        let mut hidden_states = Tensor::<P>::default(&[seq_len, self.d]);
        let mut q_buf = Tensor::<P>::default(&[seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores = Tensor::<P>::default(&[self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<P>::default(&[seq_len, self.di]);
        let mut up_buf = Tensor::<P>::default(&[seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

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
                let k = &mut cache.k_cache_within(layer, past_seq_len..total_seq_len); // (seq, n_kv_h * dqkv)
                let v = &mut cache.v_cache_within(layer, past_seq_len..total_seq_len); // (seq, n_kv_h * dqkv)
                OP::matmul_transb(
                    q,
                    P::zero(),
                    &hidden_states,
                    &self.params.wq[layer],
                    P::one(),
                );
                OP::matmul_transb(
                    k,
                    P::zero(),
                    &hidden_states,
                    &self.params.wk[layer],
                    P::one(),
                );
                OP::matmul_transb(
                    v,
                    P::zero(),
                    &hidden_states,
                    &self.params.wv[layer],
                    P::one(),
                );
                OP::rope(
                    q.reshape(&[seq_len, self.n_q_h, self.dqkv]),
                    past_seq_len,
                    self.rope_theta,
                );
                OP::rope(
                    k.reshape(&[seq_len, self.n_kv_h, self.dqkv]),
                    past_seq_len,
                    self.rope_theta,
                );

                let full_k = cache.full_k_cache(layer); // (total_seq, n_kv_h * dqkv)
                let full_v = cache.full_v_cache(layer); // (total_seq, n_kv_h * dqkv)

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

        logits
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
            let q_vec = q.slice(
                Tensor::<P>::index_to_offset(&[seq_idx, kv_idx, g_idx, 0], q.shape()),
                &[dqkv],
            );
            let k_vec = k.slice(
                Tensor::<P>::index_to_offset(&[tseq_idx, kv_idx, 0], k.shape()),
                &[dqkv],
            );
            let att_score = OP::dot(&q_vec, &k_vec) / dqkv_root;
            let mut a = att_scores.slice(
                Tensor::<P>::index_to_offset(
                    &[kv_idx, g_idx, seq_idx, tseq_idx],
                    att_scores.shape(),
                ),
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

    cartesian_product2(0..n_groups, 0..seq_len).for_each(|(g_idx, seq_idx)| {
        cartesian_product2(0..n_kv_h, 0..total_seq_len).for_each(|(kv_idx, tseq_idx)| {
            let v_vec = v.slice(
                Tensor::<P>::index_to_offset(&[tseq_idx, kv_idx, 0], v.shape()),
                &[dqkv],
            );
            let att_score = att_scores.data_at(&[kv_idx, g_idx, seq_idx, tseq_idx]);
            let mut hidden_vec = hidden.slice(
                Tensor::<P>::index_to_offset(&[seq_idx, kv_idx, g_idx, 0], hidden.shape()),
                &[dqkv],
            );
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
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
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
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    if !model_dir.exists() {
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
        model.params.embedding_table.data_at(&[31012, 55]),
        &-0.009104947,
        1e-6
    ));
    assert!(float_eq(
        model.params.lm_head.data_at(&[20100, 3]),
        &-0.032863498,
        1e-6
    ));
}
