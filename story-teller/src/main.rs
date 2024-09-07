use clap::Parser;
use lm_infer_core::{
    model::Llama,
    session::{LmSession, TokenGeneration},
};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The path to the model directory
    #[arg(short, long)]
    model_dir: String,
}

fn main() {
    let args = Args::parse();
    let model_dir = std::env::current_dir().unwrap().join(args.model_dir);
    assert!(
        model_dir.exists(),
        "Model directory not found: {}",
        model_dir.display()
    );

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
