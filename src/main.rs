mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time, ";
    print!("\n{input}");
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    let output_ids = llama.generate(input_ids, 500, 0.9, 4, 1.);
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());

    #[cfg(feature = "perf")]
    {
        if output_ids.len() > 1 {
            println!("--- Performance ---");
            let perf_info = llama.perf_info().lock().unwrap();
            let token_per_sec =
                input_ids.len() as f64 / perf_info.prompt_duration().unwrap().as_secs_f64();
            println!(
                "Prompt Processing(prompt token len: {}): {:.6} tokens/s",
                input_ids.len(),
                token_per_sec
            );

            let token_per_sec = output_ids.len() as f64
                - 1. / {
                    (perf_info.total_generation_duration().unwrap()
                        - perf_info.prompt_duration().unwrap())
                    .as_secs_f64()
                };
            println!("Story Generation: {:.6} tokens/s", token_per_sec);
        } else {
            println!("Output is too short to measure performance.");
        }
    }
}
