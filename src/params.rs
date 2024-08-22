use crate::{config::LlamaConfigJson, tensor::Tensor};
use num_traits::Num;
use safetensors::SafeTensors;

pub struct LLamaParams<T: Num> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).unwrap();
            match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    // from litte endian data (`_tobytes`)
                    let data_from_bytes = tensor_view
                        .data()
                        .chunks_exact(core::mem::size_of::<f32>())
                        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()));
                    Tensor::new(data_from_bytes.collect(), tensor_view.shape())
                }
                _ => unimplemented!(),
            }
        };

        macro_rules! get_tensor_vec {
            ($name_pattern:literal) => {
                (0..config.num_hidden_layers)
                    .map(|i| get_tensor(&format!($name_pattern, i)))
                    .collect()
            };
        }
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),

            rms_att_w: get_tensor_vec!("model.layers.{}.input_layernorm.weight"),
            wq: get_tensor_vec!("model.layers.{}.self_attn.q_proj.weight"),
            wk: get_tensor_vec!("model.layers.{}.self_attn.k_proj.weight"),
            wv: get_tensor_vec!("model.layers.{}.self_attn.v_proj.weight"),
            wo: get_tensor_vec!("model.layers.{}.self_attn.o_proj.weight"),

            rms_ffn_w: get_tensor_vec!("model.layers.{}.post_attention_layernorm.weight"),
            w_up: get_tensor_vec!("model.layers.{}.mlp.up_proj.weight"),
            w_gate: get_tensor_vec!("model.layers.{}.mlp.gate_proj.weight"),
            w_down: get_tensor_vec!("model.layers.{}.mlp.down_proj.weight"),

            lm_head: get_tensor("lm_head.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
        }
    }
}
