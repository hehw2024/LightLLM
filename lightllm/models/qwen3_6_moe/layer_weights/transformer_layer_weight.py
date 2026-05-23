from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)


class Qwen36MOETransformerLayerWeight(Qwen35MOETransformerLayerWeight):
    """Layer weight for Qwen3.6 MoE layers.

    Inherits the fused gate_up_proj/down_proj splitting logic from Qwen3.5 MoE.
    Weight naming and tensor parallelism are identical.
    """

    pass
