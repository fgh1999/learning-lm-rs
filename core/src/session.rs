use crate::{kvcache::KVCache, model::LmModel, operators as OP, tensor::Tensor};
use getset::Getters;
use num_traits::{Float, Num};
use std::{
    ops::{AddAssign, DivAssign, MulAssign},
    sync::Arc,
    time::Duration,
};

#[derive(Getters)]
pub struct LmSession<P: Float + Default + Copy, TID: Num + Copy, M: LmModel<TID, P>> {
    kv_cache: KVCache<TID, P>,
    model: Arc<M>,
    #[getset(get = "pub")]
    records: Vec<PerfTimeRecord>,
}

#[derive(Default, Debug)]
pub struct PerfTimeRecord {
    /// Total duration of generation
    pub total_generation_duration: Duration,
    /// Duration of processing prompt tokens
    pub prompt_duration: Duration,
    pub prompt_token_num: usize,
    pub output_token_num: usize,
}

impl PerfTimeRecord {
    pub fn duration_after_prompt(&self) -> Duration {
        self.total_generation_duration - self.prompt_duration
    }

    /// token/s
    pub fn prompt_prossing_performance(&self) -> f64 {
        self.prompt_token_num as f64 / self.prompt_duration.as_secs_f64()
    }

    /// Performance of generating a token after the prompt processing in token/s
    pub fn generation_performance(&self) -> Option<f64> {
        if self.output_token_num < 1 {
            return None;
        }
        Some((self.output_token_num - 1) as f64 / self.duration_after_prompt().as_secs_f64())
    }
}

impl<P: Float + Default + Copy, TID: Num + Copy, M: LmModel<TID, P>> LmSession<P, TID, M> {
    pub fn new(model: Arc<M>) -> Self {
        LmSession {
            kv_cache: KVCache::new(model.layer_num(), model.max_seq_len(), model.kv_dim()),
            model,
            records: Vec::new(),
        }
    }

    pub fn avg_performance(&self) -> (Option<f64>, Option<f64>) {
        if self.records.is_empty() {
            return (None, None);
        }

        let avg_prompt_token_per_sec = self
            .records
            .iter()
            .map(|record| record.prompt_prossing_performance())
            .sum::<f64>()
            / self.records.len() as f64;

        let token_per_sec_after_prompt = self
            .records
            .iter()
            .filter_map(|record| record.generation_performance())
            .collect::<Vec<_>>();
        let avg_token_per_sec = if !token_per_sec_after_prompt.is_empty() {
            let num = token_per_sec_after_prompt.len();
            Some(token_per_sec_after_prompt.into_iter().sum::<f64>() / num as f64)
        } else {
            None
        };

        (Some(avg_prompt_token_per_sec), avg_token_per_sec)
    }

    pub fn print_perf_info(&self) {
        println!("--- Performance ---");
        let (avg_prompt_token_per_sec, avg_token_per_sec) = self.avg_performance();
        if let Some(avg_prompt_token_per_sec) = avg_prompt_token_per_sec {
            println!(
                "AVG Prompt Processing: {:.6} tokens/s",
                avg_prompt_token_per_sec
            );
        }
        if let Some(avg_token_per_sec) = avg_token_per_sec {
            println!("AVG Generation: {:.6} tokens/s", avg_token_per_sec);
        } else {
            println!("Output is too short to measure performance.");
        }
    }
}

pub trait TokenGeneration {
    fn generate(
        &mut self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        repetition_penalty: Option<f32>,
    ) -> Vec<u32>;

    /// Reverts the session to the state before the ith generation.
    /// Returns an iterator of all the token ids after this revertion.
    fn revert_to_before(&mut self, ith: usize) -> impl Iterator<Item = u32>;

    fn generation_records(&self) -> &[PerfTimeRecord];
}
impl<
        P: Float + std::iter::Sum + Sync + Send + MulAssign + DivAssign + AddAssign + Copy + Default,
        M: LmModel<u32, P>,
    > TokenGeneration for LmSession<P, u32, M>
{
    fn generate(
        &mut self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        repetition_penalty: Option<f32>,
    ) -> Vec<u32> {
        let max_len = max_len.min(self.model.max_seq_len() - token_ids.len());
        assert!(repetition_penalty.map(|r| r > 0.).unwrap_or(true));
        let mut result = Vec::<u32>::new();

        macro_rules! generate_next_token_id {
            ($prompt_token_ids:expr) => {{
                let token_ids_len = $prompt_token_ids.len();
                let token_tensor = Tensor::<u32>::new($prompt_token_ids, &[1, token_ids_len]);
                let mut logits = self.model.forward(&token_tensor, &mut self.kv_cache);
                if let Some(repetition_penalty) = repetition_penalty {
                    OP::repetition_penalty(&mut logits, repetition_penalty, token_tensor.data())
                }
                OP::random_sample(&logits, top_p, top_k, temperature)
            }};
        }

        #[cfg(feature = "perf")]
        let start = std::time::Instant::now();
        #[allow(unused_variables)]
        let first_iter_duration = Duration::default();

        result.push({
            let prompt_token_ids = if !token_ids.is_empty() {
                token_ids.to_vec()
            } else {
                vec![self.model.bos_token_id()]
            };
            generate_next_token_id!(prompt_token_ids)
        });

        #[cfg(feature = "perf")]
        let first_iter_duration = start.elapsed();

        while result.len() < max_len && result.last() != Some(&self.model.eos_token_id()) {
            result.push(generate_next_token_id!(vec![*result.last().unwrap()]));
        }

        #[allow(unused_variables)]
        let total_generation_duration = Duration::default();
        #[cfg(feature = "perf")]
        let total_generation_duration = start.elapsed();
        let record = PerfTimeRecord {
            total_generation_duration,
            prompt_duration: first_iter_duration,
            prompt_token_num: token_ids.len(),
            output_token_num: result.len(),
        };
        self.records.push(record);

        result
    }

    fn revert_to_before(&mut self, ith: usize) -> impl Iterator<Item = u32> {
        self.records.truncate(ith);
        self.kv_cache.truncate(
            self.records
                .last()
                .map(|record| record.prompt_token_num + record.output_token_num)
                .unwrap_or(0),
        );
        self.kv_cache.token_ids()
    }

    fn generation_records(&self) -> &[PerfTimeRecord] {
        &self.records
    }
}
