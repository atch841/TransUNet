import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from inference import inference

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, LiTS_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_train = LiTS_dataset(base_dir='/home/viplab/data/train5/', split='train', transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                   tumor_only=True, pseudo=True)
    db_train_1p = LiTS_dataset(base_dir='/home/viplab/data/train5_1p_half/', split='train', transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                   tumor_only=True)
    

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    trainloader_1p = DataLoader(db_train_1p, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    if args.unfreeze_epoch:
        model.freeze_backbone = True
    ce_loss = CrossEntropyLoss(weight=torch.tensor([1.0, 389.0]).cuda(), ignore_index=255)
    # dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr * args.pseudo_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_1p = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    iter_num_1p = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    max_iterations_1p = args.max_epochs * len(trainloader_1p)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    inference(args, model, -1)
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        if args.model == 'deeplab_resnest':
            model.mode = 'TRAIN'
        iter_num = train_epoch(trainloader, args, model, ce_loss, optimizer, base_lr * args.pseudo_lr, 
                        iter_num, max_iterations, writer, epoch_num)
        iter_num_1p = train_epoch(trainloader_1p, args, model, ce_loss, optimizer_1p, base_lr, 
                        iter_num_1p, max_iterations_1p, writer, epoch_num)
            

        eval_interval = 5
        if (epoch_num + 1) % eval_interval == 0:
            tumor_dice = inference(args, model, epoch_num + 1)
            model.train()
            if args.model == 'deeplab_resnest':
                model.mode = 'TRAIN'
            writer.add_scalar('info/tumor_dice', tumor_dice, iter_num)
            if tumor_dice > best_performance:
                best_performance = tumor_dice
                save_mode_path = os.path.join(snapshot_path, 'best_model_ep' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            if args.pretrain_epoch != -1:
                logdir = snapshot_path[:snapshot_path.rfind('/')+1]
                with open(logdir + 'log_all.txt', "a") as logfile:
                    logfile.write(f'{args.pretrain_epoch}: {best_performance}\n')
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def train_epoch(trainloader, args, model, ce_loss, optimizer, base_lr, iter_num,
             max_iterations, writer, epoch_num):
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        aux_outputs = None
        if args.model == 'deeplab_resnest':
            outputs, aux_outputs = model(image_batch)
        else:
            outputs = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch[:].long())
        # if args.dataset == 'LiTS_tumor':
        #     loss_dice = dice_loss(outputs, label_batch, weight=[1, 1], softmax=True)
        # else:
        #     loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = loss_ce #+ 0.5 * loss_dice
        if aux_outputs != None:
            loss_ce_aux = ce_loss(aux_outputs, label_batch[:].long())
            # loss_dice_aux = dice_loss(aux_outputs, label_batch, softmax=True)
            loss += 0.4 * loss_ce_aux 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)
        writer.add_scalar('info/loss_ce', loss_ce, iter_num)

        logging.info('epoch %d iteration %d : loss : %f, loss_ce: %f' % (epoch_num, iter_num, loss.item(), loss_ce.item()))
    return iter_num