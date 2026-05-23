from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_6.model import Qwen3_6TpPartModel
from lightllm.models.qwen3_6_moe.layer_weights.transformer_layer_weight import (
    Qwen36MOETransformerLayerWeight,
)


@ModelRegistry("qwen3_6_moe", is_multimodal=True)
class Qwen3_6MOETpPartModel(Qwen3_6TpPartModel):
    """
    Qwen3.6 Multimodal MoE Model

    Inherits from Qwen3.6 dense with MoE weight loading:
    - 256 experts, 8 routed + 1 shared per token
    - moe_intermediate_size=512 per expert
    - Same hybrid attention as dense variant

    Compatible with Qwen3.6-35B-A3B and similar MoE checkpoints.
    """

    transformer_weight_class = Qwen36MOETransformerLayerWeight
