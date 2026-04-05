# LLM Cache Directory

This directory stores the quantized Llama-3 8B model for vLLM inference.

## Setup Instructions

1. Download the model:
```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir llm_cache/llama3_8b
```

2. Or use a quantized version:
```bash
huggingface-cli download TheBloke/Llama-3-8B-Instruct-AWQ --local-dir llm_cache/llama3_8b_awq
```

## Configuration

The vLLMInferenceWrapper will automatically detect the model format.

```python
from core.vllm_inference_wrapper import vLLMInferenceWrapper

llm = vLLMInferenceWrapper(
    model_path='models/llm_cache/llama3_8b_awq',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5
)
```

## Model Requirements

- ~16GB VRAM for full precision
- ~8GB VRAM for AWQ quantized
- RTX 5090 recommended for production
