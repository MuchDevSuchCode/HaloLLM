use axum::{routing::post, Json, Router};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    model_path: String,
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
    duration_ms: u64,
}

async fn generate(Json(payload): Json<GenerateRequest>) -> Json<GenerateResponse> {
    let start = std::time::Instant::now();
    
    // Initialize CUDA device, fallback to CPU
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    
    // Load tokenizer
    let tokenizer_path = std::path::Path::new("tokenizer.json"); 
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .expect("Failed to load tokenizer.json. Please ensure it exists in the current directory.");
    
    // Open and parse GGUF model
    let mut file = std::fs::File::open(&payload.model_path)
        .expect(&format!("Failed to open model at {}", payload.model_path));
    let model = gguf_file::Content::read(&mut file).expect("Failed to read GGUF content");
    let mut model_weights = ModelWeights::from_gguf(model, &mut file, &device)
        .expect("Failed to load model weights");
    
    // Tokenize prompt
    let tokens = tokenizer
        .encode(payload.prompt.clone(), true)
        .expect("Failed to encode prompt")
        .get_ids()
        .to_vec();
    
    let mut generated_tokens = vec![];
    let mut next_tokens = tokens.clone();
    let mut index_pos = 0;
    
    // Text generation loop
    let max_len = 100;
    for _ in 0..max_len {
        let input = Tensor::new(next_tokens.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = model_weights.forward(&input, index_pos).unwrap();
        
        let mut logits_last = logits;
        if logits_last.dim(0).unwrap() == 1 {
            logits_last = logits_last.squeeze(0).unwrap();
        }
        
        if logits_last.rank() == 2 {
            let seq_len = logits_last.dim(0).unwrap();
            logits_last = logits_last.get(seq_len - 1).unwrap();
        }
        
        let next_token = logits_last.argmax(0).unwrap().to_scalar::<u32>().unwrap();
        generated_tokens.push(next_token);
        
        index_pos += next_tokens.len();
        next_tokens = vec![next_token];
        
        if next_token == 2 { // Assume 2 is EOS for Llama and related models
            break;
        }
    }
    
    let generated_text = tokenizer
        .decode(&generated_tokens, true)
        .expect("Failed to decode tokens");
    
    Json(GenerateResponse { 
        text: generated_text,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/api/generate", post(generate));
    let addr = SocketAddr::from(([0, 0, 0, 0], 11435)); // Running on a different port than Ollama
    
    println!("🚀 HaloLLM Backend booting up on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}