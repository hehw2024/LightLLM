from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.models.qwen3_5.layer_weights.pre_and_post_layer_weight import (
    Qwen35PreAndPostLayerWeight,
)
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import (
    Qwen35TransformerLayerInfer,
)
from lightllm.models.qwen3_5.infer_struct import Qwen35InferStateInfo
from lightllm.models.qwen3_5.model import QWen3_5Tokenizer
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import (
    Qwen3VLMultimodalPreLayerInfer,
)


@ModelRegistry(["qwen3_6"], is_multimodal=True)
class Qwen3_6TpPartModel(Qwen3_5TpPartModel):
    """
    Qwen3.6 Multimodal Model (Dense Variant)

    Inherits from Qwen3.5 with the same hybrid attention architecture:
    - Gated Delta Networks (linear attention) + Full Attention on alternating layers
    - Attention output gating (attn_output_gate)
    - Multimodal support (vision encoder)
    - Dense MLP layers (non-MoE)

    Key differences from Qwen3.5:
    - Larger vocabulary (248,320 tokens)
    - partial_rotary_factor=0.25 (only 25% of head_dim gets RoPE)
    - head_dim=256, larger attention dimensions
    - MTP support (mtp_num_hidden_layers=1)
    - Explicit layer_types array for hybrid attention scheduling

    Compatible with Qwen3.6-27B and similar dense checkpoints.
    """

    transformer_weight_class = Qwen35TransformerLayerWeight
    pre_and_post_weight_class = Qwen35PreAndPostLayerWeight
    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen35TransformerLayerInfer
    infer_state_class = Qwen35InferStateInfo


class QWen3_6Tokenizer(QWen3_5Tokenizer):
    """Tokenizer for Qwen3.6 multimodal model. Inherits Qwen3.5 tokenizer logic."""

    pass
