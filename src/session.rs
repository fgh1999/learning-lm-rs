use crate::{kvcache::KVCache, model::LmModel, operators as OP, tensor::Tensor};
use getset::Getters;
use num_traits::{float::TotalOrder, Float};
use std::{
    ops::{AddAssign, DivAssign, MulAssign},
    sync::Mutex,
    time::Duration,
};

#[derive(Getters)]
pub struct LmSession<T: Float + Default + Copy, M: LmModel<T>> {
    kv_cache: KVCache<T>,
    model: M,
    #[getset(get = "pub")]
    perf_info: Mutex<PerfInfo>,
}

#[derive(Getters, Default, Debug)]
pub struct PerfInfo {
    /// Total duration of generation
    #[getset(get = "pub")]
    total_generation_duration: Option<Duration>,
    /// Duration of processing prompt tokens
    #[getset(get = "pub")]
    prompt_duration: Option<Duration>,
}

impl PerfInfo {
    pub fn duration_after_prompt(&self) -> Option<Duration> {
        match (self.total_generation_duration, self.prompt_duration) {
            (Some(total), Some(prompt)) => Some(total - prompt),
            _ => None,
        }
    }

    /// token/s
    pub fn prompt_prossing_performance(&self, prompt_token_len: usize) -> Option<f64> {
        self.prompt_duration.map(|d| prompt_token_len as f64 / d.as_secs_f64())
    }

    /// Performance of generating a token after the prompt processing in token/s
    pub fn generation_performance(&self, output_token_len: usize) -> Option<f64> {
        if output_token_len < 2 {
            return None;
        }
        self.duration_after_prompt()
            .map(|d| (output_token_len as f64 - 1.) / d.as_secs_f64())
    }
}

impl<T: Float + Default + Copy, M: LmModel<T>> LmSession<T, M> {
    pub fn new(model: M) -> Self {
        LmSession {
            kv_cache: KVCache::new(model.layer_num(), model.max_seq_len(), model.kv_dim()),
            model,
            perf_info: Mutex::new(PerfInfo::default()),
        }
    }

    pub fn print_perf_info(&self, input_len: usize, output_len: usize) {
        println!("--- Performance ---");
        let perf_info = self.perf_info.lock().unwrap();
        let token_per_sec = perf_info.prompt_prossing_performance(input_len).unwrap();
        println!(
            "Prompt Processing(prompt token len: {}): {:.6} tokens/s",
            input_len, token_per_sec
        );

        let Some(token_per_sec) = perf_info.generation_performance(output_len) else {
            println!("Output is too short to measure performance.");
            return;
        };
        println!("Generation: {:.6} tokens/s", token_per_sec);
    }
}

impl<
        T: Float
            + std::iter::Sum
            + Sync
            + Send
            + MulAssign
            + DivAssign
            + AddAssign
            + Copy
            + Clone
            + Default
            + TotalOrder,
        M: LmModel<T>,
    > LmSession<T, M>
{
    pub fn generate(
        &mut self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let max_len = max_len.min(self.model.max_seq_len() - token_ids.len());
        debug_assert!(max_len > 0);

        let mut result = Vec::<u32>::new();

        macro_rules! generate_next {
            ($prompt_token_ids:expr) => {{
                let token_ids_len = $prompt_token_ids.len();
                let token_tensor = Tensor::<u32>::new($prompt_token_ids, &[1, token_ids_len]);
                let logits = self.model.forward(&token_tensor, &mut self.kv_cache);
                OP::random_sample(&logits, top_p, top_k, temperature)
            }};
        }

        #[cfg(feature = "perf")]
        let start = std::time::Instant::now();

        result.push({
            let prompt_token_ids = if !token_ids.is_empty() {
                token_ids.to_vec()
            } else {
                vec![self.model.bos_token_id()]
            };
            generate_next!(prompt_token_ids)
        });

        #[cfg(feature = "perf")]
        let first_iter_duration = start.elapsed();

        while result.len() < max_len && result.last() != Some(&self.model.eos_token_id()) {
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
