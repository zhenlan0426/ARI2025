"""
2025.3.17
2025.3.19
4.50.0.dev0
0.15.2
__UNSLOTH_VERSIONING__
"""

torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False}
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from peft.tuners.lora.tp_layer import (Any, __name__, nn, torch)


torch_addmm = torch.addmm
torch_add   = torch.add
# @torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
    xA = dropout(x) @ lora_A.weight.t()
    # output = result + scaling * xA @ lora_B.weight.t()
    shape = result.shape
    output = torch_addmm(
        result.view(-1, shape[-1]),
        xA.view(-1, xA.shape[-1]),
        lora_B.weight.t(),
        alpha = scaling,
        beta = 1,
    ).view(shape)

    bias = lora_B.bias
    if bias is not None:
        output = torch_add(
        output,
        bias,
        alpha = scaling,
    )
    return output
pass

def unsloth_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    
    adapter_names = kwargs.pop("adapter_names", None)
    # If weight is used for matrix multiplication here, the final aggregation operation of the original
    # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
    # output of the original parallel_linear layer.
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result, bias = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")
    elif self.merged:
        result, bias = self.base_layer(x, *args, **kwargs)
    else:
        result, bias = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            if not torch.is_autocast_enabled(): result, x = result.to(lora_A.weight.dtype), x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                return lora_forward(result, lora_A, lora_B, dropout, x, scaling)
            else:
                if isinstance(dropout, torch.nn.Identity) or not self.training:
                    base_result = result
                else:
                    x = dropout(x)
                    base_result = None

                result = result + self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=self.get_base_layer(),
                    base_result=base_result,
                )

        result = result.to(torch_result_dtype)
    return result, bias
