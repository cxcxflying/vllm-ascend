import torch
from torch.nn.parameter import Parameter

ASCEND_ALIGN_BYTES = 512
BF16_FP16_BYTES = 2


def ascend_unquant_linear_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    weight = layer.weight
    if weight.dtype == torch.float16 or weight.dtype == torch.bfloat16:
        if (weight.shape[0] % (ASCEND_ALIGN_BYTES / BF16_FP16_BYTES) == 0 and
                weight.shape[1] % (ASCEND_ALIGN_BYTES / BF16_FP16_BYTES) != 0):
            layer.weight = Parameter(weight.t().contiguous().t(), requires_grad=False)

from vllm.model_executor.layers import linear
linear.UnquantizedLinearMethod.process_weights_after_loading = ascend_unquant_linear_process_weights_after_loading