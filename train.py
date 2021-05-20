import argparse
from networks.deeplab import DeepLab
from networks.denseunet import DenseUNet
from datasets.dataset_synapse import LiTS_dataset, LiTS_tumor_dataset
import logging
from networks.unet_model import UNet
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,#
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,#
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,#
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--model', type=str, default='TU', choices=['TU', 'UNet', 'denseunet', 'deeplab', 'deeplab_xception'],
                    help='model to use')#
parser.add_argument('--is_pretrain', type=str, default='',
                    help='pretrain model path')#
parser.add_argument('--pretrain_epoch', type=int, default=-1,
                    help='epoch of loaded pretrained model')
parser.add_argument('--unfreeze_epoch', type=int, default=0,
                    help='epoch to unfreeze backbone')
parser.add_argument('--pretrain_folder', type=str, default='',
                    help='pretrained folder name')
args = parser.parse_args()


if __name__ == "__main__":
    print("TRAIN")
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'LiTS': {
            'Dataset': LiTS_dataset,
            'root_path': '/home/viplab/data/train5/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 3,
            'z_spacing': 1,
        },
        'LiTS_tumor': {
            'Dataset': LiTS_tumor_dataset,
            'root_path': '/home/viplab/data/train5/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'LiTS_tumor_50p': {
            'Dataset': LiTS_tumor_dataset,
            'root_path': '/home/viplab/data/train5_50p/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'LiTS_tumor_20p': {
            'Dataset': LiTS_tumor_dataset,
            'root_path': '/home/viplab/data/train5_20p/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'LiTS_tumor_10p': {
            'Dataset': LiTS_tumor_dataset,
            'root_path': '/home/viplab/data/train5_10p/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'LiTS_tumor_1p': {
            'Dataset': LiTS_tumor_dataset,
            'root_path': '/home/viplab/data/train5_1p/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'LiTS_tumor_5p_half': {
            'Dataset': LiTS_tumor_dataset,
            'root_path': '/home/viplab/data/train5_5p_half/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'LiTS_tumor_1p_half': {
            'Dataset': LiTS_tumor_dataset,
            'root_path': '/home/viplab/data/train5_1p_half/',
            'volume_path': '/home/viplab/data/stage1/test/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    # args.is_pretrain = True
    args.exp = '{}_'.format(args.model) + dataset_name + str(args.img_size)
    snapshot_path = '/home/viplab/data/model/'
    snapshot_path = snapshot_path + '{}/'.format(args.pretrain_folder) if args.pretrain_folder else snapshot_path
    snapshot_path += "{}/{}".format(args.exp, args.model)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_path = snapshot_path + '_pe'+str(args.pretrain_epoch) if args.pretrain_epoch != -1 else snapshot_path
    snapshot_path = snapshot_path + '_ue'+str(args.unfreeze_epoch) if args.unfreeze_epoch else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if args.model == 'TU':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        if args.is_pretrain == 'official':
            print('loading pretrain')
            net.load_from(weights=np.load(config_vit.pretrained_path))
        elif args.is_pretrain:
            print('loading pretrain:', args.is_pretrain)
            net.load_ptbb(args.is_pretrain)
    elif args.model == 'UNet':
        net = UNet(1, args.num_classes).cuda()
        if args.is_pretrain:
            print('loading pretrain')
            state_dict = torch.load('/home/viplab/data/unet_carvana_scale1_epoch5.pth')
            model_dict = net.state_dict()
            pretrained_state = { k:v for k,v in state_dict.items() \
                                if k in model_dict and v.size() == model_dict[k].size() }
            model_dict.update(pretrained_state)
            net.load_state_dict(model_dict)
    elif args.model == 'denseunet':
        if args.is_pretrain:
            net = DenseUNet(args.num_classes, pretrained_encoder_uri='https://download.pytorch.org/models/densenet121-a639ec97.pth').cuda()
        else:
            net = DenseUNet(args.num_classes).cuda()
    elif args.model == 'deeplab':
        net = DeepLab(sync_bn=False, num_classes=args.num_classes).cuda()
    elif args.model == 'deeplab_xception':
        net = DeepLab(sync_bn=False, num_classes=args.num_classes, backbone='xception', output_stride=8).cuda()
    else:
        raise NotImplementedError('model not found!')

    # trainer = {'Synapse': trainer_synapse, 'LiTS': trainer_synapse, 'LiTS_tumor': trainer_synapse}
    # trainer[dataset_name](args, net, snapshot_path)
    trainer_synapse(args, net, snapshot_path)