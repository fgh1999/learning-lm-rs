use lm_infer::{
    model::Llama,
    session::{Generation, LmSession},
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");

    let llama = Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut sess = LmSession::new(llama.into());

    let input = "Once upon a time, ";
    print!("\n{input}");

    let input_ids = {
        let binding = tokenizer.encode(input, true).unwrap();
        binding.get_ids().to_owned()
    };
    let output_ids = sess.generate(&input_ids, 500, 0.9, 4, 1., None);
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());

    #[cfg(feature = "perf")]
    sess.print_perf_info();
}
