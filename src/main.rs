use axum::{routing::post, Json, Router};
use llama_cpp_rs::{options::{ModelOptions, PredictOptions}, LLama};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

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
    
    // Configure hardware layers to use the Vulkan API fully 
    // This tells llama.cpp to offload all calculations internally!
    let model_options = ModelOptions {
        n_gpu_layers: 99,
        ..Default::default()
    };
    
    // Load native model, automatically handling embedded GGUF tokenizer parsing
    let llama = match LLama::new(payload.model_path.clone(), &model_options) {
        Ok(l) => l,
        Err(e) => {
            return Json(GenerateResponse {
                text: format!("Failed to load model: {:?}", e),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }
    };
    
    // Setup prediction loop (equivalent to tensor forward-pass, argmax, and sequence decoding!)
    let predict_options = PredictOptions {
        n_predict: 100, // Generate up to 100 tokens max
        ..Default::default()
    };
    
    let generated_text = match llama.predict(payload.prompt.clone(), predict_options) {
        Ok(text) => text,
        Err(e) => format!("Inference failed: {:?}", e),
    };
    
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