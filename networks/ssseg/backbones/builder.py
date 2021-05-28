'''
Function:
    build normalization
Function:
    build activation functions
Function:
    build the backbone
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
# from .hardswish import HardSwish
# from .hardsigmoid import HardSigmoid 
# from .layernorm import LayerNorm
# from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
# from .syncbatchnorm import MMCVSyncBatchNorm, TORCHSyncBatchNorm

'''layer normalization'''
LayerNorm = nn.LayerNorm
'''batchnorm1d'''
BatchNorm1d = nn.BatchNorm1d
'''batchnorm2d'''
BatchNorm2d = nn.BatchNorm2d
'''batchnorm3d'''
BatchNorm3d = nn.BatchNorm3d

class groupnorm_wrapper(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(1, channels)
    def forward(self, x):
        x = self.gn(x)
        return x

GroupNorm = groupnorm_wrapper

'''build normalization'''
def BuildNormalization(norm_type='batchnorm2d', instanced_params=(0, {}), only_get_all_supported=False, **kwargs):
    supported_dict = {
        'layernorm': LayerNorm,
        'batchnorm1d': BatchNorm1d,
        'batchnorm2d': BatchNorm2d,
        'batchnorm3d': BatchNorm3d,
        'groupnorm': GroupNorm,
        # 'syncbatchnorm': TORCHSyncBatchNorm,
        # 'syncbatchnorm_mmcv': MMCVSyncBatchNorm,
    }
    if only_get_all_supported: return list(supported_dict.values())
    assert norm_type in supported_dict, 'unsupport norm_type %s...' % norm_type
    norm_layer = supported_dict[norm_type](instanced_params[0], **instanced_params[1])
    if norm_type in ['syncbatchnorm_mmcv']: norm_layer._specify_ddp_gpu_num(1)
    return norm_layer

'''build activation functions'''
def BuildActivation(activation_type, **kwargs):
    supported_activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'relu6': nn.ReLU6,
        'sigmoid': nn.Sigmoid,
        # 'hardswish': HardSwish,
        'leakyrelu': nn.LeakyReLU,
        # 'hardsigmoid': HardSigmoid,
    }
    assert activation_type in supported_activations, 'unsupport activation type %s...' % activation_type
    return supported_activations[activation_type](**kwargs)


