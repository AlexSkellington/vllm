# Quantization

Quantization trades off model precision for smaller memory footprint, allowing large models to be run on a wider range of devices.

!!! tip
    To get started with quantization, see [LLM Compressor](llm_compressor/README.md), a library for optimizing models for deployment with vLLM that supports FP8, INT8, INT4, and other quantization formats.

The following are the supported quantization formats for vLLM:

- [AutoAWQ](auto_awq.md)
- [BitsAndBytes](bnb.md)
- [GPTQModel](gptqmodel.md)
- [Intel Neural Compressor](inc.md)
- [LLM Compressor](llm_compressor/README.md)
    - [FP8 W8A8](llm_compressor/fp8.md)
    - [INT4 W4A16](llm_compressor/int4.md)
    - [INT8 W4A8](llm_compressor/int8_w4a8.md)
    - [INT8 W8A8](llm_compressor/int8_w8a8.md)
- [NVIDIA Model Optimizer](modelopt.md)
- [Online Quantization](online.md)
- [AMD Quark](quark.md)
- [Quantized KV Cache](quantized_kvcache.md)
- [TorchAO](torchao.md)
- [FP8 ViT Encoder Attention](fp8_vit_attn.md)

## Selecting Linear Backends per Quantization

`--linear-backend` selects one backend for all quantized linear layers. For
mixed-precision models that use more than one linear quantization scheme, use
`linear_backend_per_quant` to override the backend for individual schemes:

```bash
vllm serve <model> \
  --linear-backend cutlass \
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"humming"}}'
```

Here, NVFP4 W4A16 linear layers use Humming, while all other quantized linear
layers use CUTLASS. Per-quantization overrides take precedence over
`--linear-backend`; schemes without an override continue to use the global
setting, including automatic selection when it is `auto`.

## Alex_C.T integer linears on RTX 3090

The lab Humming fork provides SM86 integer Tensor Core kernels for packed
compressed-tensors W4A4, W4A8, W5A8, W6A8, and W8A8 linear layers. W4A4 uses
INT4 MMA. The INT8 modes consume packed weights and expand narrower weights
inside GEMM. Weight precision comes from the checkpoint.

To request integer activations for existing weight-only checkpoints, use
`VLLM_HUMMING_INPUT_QUANT_CONFIG`. This example uses group-128 INT8 activation
quantization with a grouped INT5 checkpoint.

```bash
VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8","group_size":128}' \
  vllm serve /path/to/packed-int5-model \
  --kernel-config '{"linear_backend_per_quant":{"mixed_precision":"humming"}}'
```

Use `"dtype":"int4"` for uniform four-bit activations or `"dtype":"iq4nl"` for
four-bit activations with the IQ4_NL codebook. Both work with packed INT4, INT5,
INT6, INT8, IQ4_NL, and IQ4_XS weights. Symmetric W4A4 uses native INT4 MMA
where the scale groups permit it. The other four-bit modes pass the selected
activation levels as signed bytes to INT8 MMA. A16 uses the model's FP16 or BF16
dtype and can be selected by leaving the activation override unset.

Omit `group_size` for per-token activation scaling. Packed checkpoints can also declare dynamic
integer activation quantization in their own metadata. Incompatible shapes
or backends produce a load error for explicit integer paths. The Alex_C.T
Humming build adds SM86 schedules for benchmarked projection shapes and nested
32/64/128 symmetric scale groups for INT8 MMA.

The lab build looks up the exact signed codebook levels for IQ4_NL and IQ4_XS
weights in registers before MMA.
For IQ4_XS, the loader consumes the same indices and the scale of each
32-weight sub-block. `tools/as1/convert_iq4_tensor.py` exports those fields
from GGUF for checkpoint assembly. Use `{"dtype":"int8"}` for per-token
activation scales or `{"dtype":"int8","group_size":32}` for group scales.
The model's parameter dtype determines the serving precision of the imported
scales.

`--kv-cache-dtype int8_per_token_head` and `--kv-cache-dtype int4_per_token_head`
select the Triton attention kernels for integer KV storage. With either cache
format, `LAB_INTEGER_KV_QK=1` enables the Alex_C.T SM86 INT8 query–key path
for prefill with at least 64 query tokens and head dimensions of at least 64.
It quantizes each query row and accumulates the query–key dot product in INT32.
Softmax and value accumulation use the regular precision for that cache
format. The flag is opt-in because it changes prefill numerics. Compare these
modes against the deployed attention backend at the same context length and
speculative decoding settings before choosing a serving configuration.
`LAB_INT4_KV_SPLIT_K=1` enables split-K for small speculative verification
batches using packed INT4 KV on SM86. It parallelizes the cache scan across
segments and combines their softmax statistics before writing the output.
For 256-wide heads on SM86, `VLLM_LAB_INT8_KV_TUNED=1` enables the Alex_C.T
INT8 query–key path for split-K decode and large prefill on
`int8_per_token_head`. It also widens the KV tiles and uses more split-K
segments when few requests would otherwise underfill the GPU.

## Supported Hardware

The table below shows the compatibility of various quantization implementations with different hardware platforms in vLLM:

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| Implementation            | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| ------------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- | ------- |
| AWQ                       | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| GPTQ                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌    | ✅︎*    | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| llm-compressor INT8 (W8A8)| ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ✅︎      | ✅︎      |
| llm-compressor INT8 (W4A8)| ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ✅︎      |
| llm-compressor FP8 (W8A8) | ❌    | ❌     | ❌     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |
| bitsandbytes              | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| DeepSpeedFP               | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| GGUF                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |

- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0.
- ✅︎ indicates that the quantization method is supported on the specified hardware.
- ❌ indicates that the quantization method is not supported on the specified hardware.
- All Intel Gaudi quantization support has been migrated to [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi).
- *Turing does not support Marlin MXFP4.

!!! note
    For information on quantization support on Google TPU, please refer to the [TPU-Inference Feature Matrix](https://docs.vllm.ai/projects/tpu/en/latest/features/#quantization-support) documentation.

!!! note
    This compatibility chart is subject to change as vLLM continues to evolve and expand its support for different hardware platforms and quantization methods.

    For the most up-to-date information on hardware support and quantization methods, please refer to [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) or consult with the vLLM development team.

## Out-of-Tree Quantization Plugins

vLLM supports registering custom, out-of-tree quantization methods using the `@register_quantization_config` decorator. This allows you to implement and use your own quantization schemes without modifying the vLLM codebase.

### Registering a Custom Quantization Method

To register a custom quantization method, create a class that inherits from `QuantizationConfig` and decorate it with `@register_quantization_config`. The `get_quant_method` dispatches to the appropriate quantize method based on the layer type:

```python
import torch
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    """Custom quantization config."""

    def get_name(self) -> str:
        return "my_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability, -1 for no restriction
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Config files to search for in model directory
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MyQuantConfig":
        # Create config from model's quantization config
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Dispatch based on layer type
        # NOTE: you only need to implement methods you care about
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod()
        elif isinstance(layer, FusedMoE):
            return MyQuantMoEMethod(layer.moe_config)
        return None
```

### Required QuantizationConfig Methods

Your custom `QuantizationConfig` subclass must implement these abstract methods:

| Method | Description |
| ------ | ----------- |
| `get_name()` | Returns the name of the quantization method |
| `get_supported_act_dtypes()` | Returns list of supported activation dtypes (e.g., `torch.float16`) |
| `get_min_capability()` | Returns minimum GPU compute capability (e.g., 80 for Ampere, -1 for no restriction) |
| `get_config_filenames()` | Returns list of config filenames to search for in model directory |
| `from_config(config)` | Class method to create config from model's quantization config dict |
| `get_quant_method(layer, prefix)` | Returns the quantization method for a given layer, or `None` to skip |

### Implementing a Quantized Linear Method

For linear layers, return a `QuantizeMethodBase` subclass from `get_quant_method`. You can extend `UnquantizedLinearMethod` as a starting point:

```python
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class MyQuantLinearMethod(UnquantizedLinearMethod):
    """Custom quantization method for linear layers."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        # Create quantized weights for the layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Apply custom quantization logic here
        ...
```

### Implementing a Quantized MoE Method

For Mixture of Experts (MoE) models, return a `FusedMoEMethodBase` subclass from `get_quant_method`. You can use `UnquantizedFusedMoEMethod` to skip MoE quantization:

```python
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class MyQuantMoEMethod(FusedMoEMethodBase):
    """Custom quantization method for MoE layers."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Create quantized weights for the MoE layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        router: "FusedMoERouter",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Apply MoE computation with quantized weights
        ...

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # Return the MoE quantization configuration
        ...
```

See existing implementations like `Fp8MoEMethod` in `vllm/model_executor/layers/quantization/fp8.py` for reference.

### Using the Plugin

Once registered, you can use your custom quantization method with vLLM:

```python
# Register your quantization method (import the module containing your config)
import my_quant_plugin

from vllm import LLM

# Use the custom quantization method
llm = LLM(model="your-model", quantization="my_quant")
```

For more information on the plugin system, see the [Plugin System documentation](../../design/plugin_system.md).
