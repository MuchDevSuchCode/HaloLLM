# HaloLLM

A high-speed local AI inference engine built with Rust and the Hugging Face `candle` framework, designed to deploy `.gguf` quantized models efficiently.

## Requirements
- Rust toolchain (cargo, rustc).
- A valid Hugging Face `tokenizer.json` available in the directory where you run the server.
- A compatible `.gguf` quantized model (e.g., LLaMA architecture).

## AMD Strix Halo / Vulkan Instructions

To route tensor operations and take full advantage of the compute capabilities on AMD's Strix Halo APU architecture natively, we compile using the cross-platform Vulkan API!

```bash
# Compile the backend with absolute maximum optimizations and Vulkan GPU support
cargo build --release --features vulkan

# Run the server
./target/release/halollm
```

## API Usage

Once the server is running, it listens on port `11435`. You can send prompt requests to the `/api/generate` endpoint.

**Example Request:**
```bash
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the significance of the Strix Halo APU architecture.",
    "model_path": "./your_model.gguf"
  }'
```

**Example Response:**
```json
{
  "text": "Strix Halo introduces a massive integrated GPU...",
  "duration_ms": 1420
}
```
