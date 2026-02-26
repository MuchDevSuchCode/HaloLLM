# HaloLLM

A high-speed local AI inference engine built with Rust and native `llama.cpp` bindings, designed to deploy `.gguf` quantized models with full GPU acceleration via Vulkan.

## Requirements
- Rust toolchain (cargo, rustc).
- A compatible `.gguf` quantized model (e.g., TinyLlama, LLaMA architecture).

## Linux Prerequisites

Install the following system packages before compiling (Ubuntu/Debian):

```bash
sudo apt install -y clang libclang-dev cmake build-essential libvulkan-dev vulkan-tools glslc
```

| Package | Why it's needed |
|---|---|
| `clang` / `libclang-dev` | `bindgen` uses clang to generate Rust FFI bindings from C/C++ headers |
| `cmake` | Builds the vendored `llama.cpp` C++ source tree |
| `build-essential` | Provides `gcc`, `g++`, `make`, and standard C headers (`stdbool.h`, etc.) |
| `libvulkan-dev` | Vulkan development headers and loader library |
| `glslc` | Vulkan GLSL shader compiler — required by CMake to compile GPU compute shaders |
| `vulkan-tools` | Optional — lets you verify your GPU is visible via `vulkaninfo` |

## AMD Strix Halo / Vulkan Build

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
