
# config for model
MODEL_CFG = {
    'benchmark': True,
    'num_classes': 2,
    'align_corners': False,
    'is_multi_gpus': False,
    'type': 'deeplabv3plus',
    'distributed': {'is_on': True, 'backend': 'nccl'},
    'norm_cfg': {'type': 'batchnorm2d', 'opts': {}},
    'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    'backbone': {
        'type': 'resnest101',
        'series': 'resnest',
        'pretrained': True,
        'outstride': 8,
        'selected_indices': (0, 1, 2, 3),
    },
    'aspp': {
        'in_channels': 2048,
        'out_channels': 512,
        'dilations': [1, 12, 24, 36],
    },
    'shortcut': {
        'in_channels': 256,
        'out_channels': 48,
    },
    'decoder': {
        'in_channels': 560,
        'out_channels': 512,
        'dropout': 0.1,
    },
    'auxiliary': {
        'in_channels': 1024,
        'out_channels': 512,
        'dropout': 0.1,
    }
}