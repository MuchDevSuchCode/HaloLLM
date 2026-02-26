# HaloLLM

A high-speed local AI inference engine built with Rust and the Hugging Face `candle` framework, designed to deploy `.gguf` quantized models efficiently.

## Requirements
- Rust toolchain (cargo, rustc).
- A valid Hugging Face `tokenizer.json` available in the directory where you run the server.
- A compatible `.gguf` quantized model (e.g., LLaMA architecture).

## AMD Strix Halo Architecture Instructions

To route tensor operations and take full advantage of the compute capabilities on AMD's Strix Halo APU architecture, use the ROCm stack by exporting the following flags before compiling and executing:

```bash
# Force the ROCm stack to recognize the Strix Halo architecture
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0

# Compile the backend with absolute maximum optimizations and AMD GPU support
cargo build --release --features candle-core/rocm

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
