use crate::{
    config::LlamaConfigJson,
    kvcache::KVCache,
    operators::{self as OP, cartesian_product2},
    params::LLamaParams,
    tensor::Tensor,
};
use getset::Getters;
use num_traits::{float::TotalOrder, Float, Num};
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
    params: LLamaParams<T>, // trained weights of this model
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

impl<T: Num + Default + Copy + Clone> Llama<T> {
    pub fn new_kvcache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }
}

impl<T: Float + std::iter::Sum + Sync + Send + MulAssign + DivAssign + AddAssign + Copy + Clone + Default + TotalOrder> Llama<T> {
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = *cache.seq_len();
        *cache.seq_len_mut() += seq_len;
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&[seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&[seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&[seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<T>::default(&[self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&[seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&[seq_len, self.di]);

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
                let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
                let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
                OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
                OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
                OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
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

                let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
                let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

                self_attention(
                    &mut hidden_states,
                    &mut att_scores,
                    q,
                    full_k,
                    full_v,
                    self.n_kv_h,
                    n_groups,
                    seq_len,
                    total_seq_len,
                    self.dqkv,
                );
                OP::matmul_transb(
                    &mut residual,
                    1.,
                    &hidden_states,
                    &self.params.wo[layer],
                    1.,
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

        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &[1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &[1, self.d]);
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&[1, self.vocab]);
        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let max_len = max_len.min(self.max_seq_len - token_ids.len());
        debug_assert!(max_len > 0);

        let mut cache = self.new_kvcache();
        let mut result = Vec::<u32>::new();

        macro_rules! generate_next {
            ($prompt_token_ids:expr) => {{
                let token_ids_len = $prompt_token_ids.len();
                let token_tensor = Tensor::<u32>::new($prompt_token_ids, &[1, token_ids_len]);
                let logits = self.forward(&token_tensor, &mut cache);
                OP::random_sample(&logits, top_p, top_k, temperature)
            }};
        }

        #[cfg(feature = "perf")]
        let start = std::time::Instant::now();

        result.push({
            let prompt_token_ids = if !token_ids.is_empty() {
                token_ids.to_vec()
            } else {
                vec![self.bos_token_id]
            };
            generate_next!(prompt_token_ids)
        });

        #[cfg(feature = "perf")]
        let first_iter_duration = start.elapsed();

        while result.len() < max_len && result.last() != Some(&self.eos_token_id) {
            result.push(generate_next!(vec![*result.last().unwrap()]));
        }

        #[cfg(feature = "perf")]
        {
            let mut perf_info = self.perf_info.lock().unwrap();
            perf_info.total_generation_duration = Some(start.elapsed());
            perf_info.prompt_duration = Some(first_iter_duration);
        }

        result
    }
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

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
            params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
            perf_info: Mutex::new(PerfInfo::default()),
        }
    }
}

fn self_attention<
    P: Float + std::iter::Sum + Sync + Send + MulAssign + DivAssign + AddAssign,
>(
    hidden_states: &mut Tensor<P>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<P>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<P>,                 // (seq, n_kv_h * n_groups, dqkv)
    k: &Tensor<P>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<P>,                 // (total_seq, n_kv_h * dqkv)
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

    cartesian_product2(0..n_kv_h, 0..n_groups).for_each(|(kv_idx, g_idx)| {
        let mut att_result = {
            let start = Tensor::<P>::index_to_offset(&[kv_idx, g_idx, 0, 0], att_scores.shape());
            let shape = [seq_len, total_seq_len];
            att_scores.slice(start, &shape)
        };
        cartesian_product2(0..seq_len, 0..total_seq_len)
            // skip masked calculation
            .filter(|&(seq_idx, tseq_idx)| (total_seq_len - seq_len) + seq_idx >= tseq_idx)
            .for_each(|(seq_idx, tseq_idx)| {
                let q_vec = q.slice(
                    Tensor::<P>::index_to_offset(&[seq_idx, kv_idx, g_idx, 0], q.shape()),
                    &[dqkv],
                );
                let k_vec = k.slice(
                    Tensor::<P>::index_to_offset(&[tseq_idx, kv_idx, 0], k.shape()),
                    &[dqkv],
                );
                unsafe {
                    att_result.with_data_mut_at(&[seq_idx, tseq_idx], |_| {
                        OP::dot(&q_vec, &k_vec) / dqkv_root
                    });
                }
            });
    });
    OP::masked_softmax(att_scores);

    let v = v.slice(0, &[total_seq_len, n_kv_h, dqkv]);
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
            let hv = unsafe { hidden_vec.data_mut() };
            hv.iter_mut()
                .zip(v_vec.data())
                .for_each(|(h, &val)| *h += val * att_score);
        });
    });
}

fn mlp<P: Float + std::iter::Sum + Sync + Send + MulAssign>(
    residual: &mut Tensor<P>,
    hidden_states: &mut Tensor<P>,
    gate: &mut Tensor<P>,
    up: &mut Tensor<P>,
    w_up: &Tensor<P>,
    w_down: &Tensor<P>,
    w_gate: &Tensor<P>,
    rms_w: &Tensor<P>,
    eps: impl Float,
) {
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    OP::matmul_transb(gate, 0., hidden_states, w_gate, 1.);
    OP::matmul_transb(up, 0., hidden_states, w_up, 1.);
    OP::silu(up, gate);
    OP::matmul_transb(residual, 1., up, w_down, 1.);
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
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
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
