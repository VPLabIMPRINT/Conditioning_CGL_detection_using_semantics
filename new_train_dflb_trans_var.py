# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from config import cfg
from new_dataset import TrainDataset
from mit_semseg.models.new_models_dflb_trans_var import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
# from focal_loss.focal_loss import FocalLoss
# import numpy as np
torch.set_printoptions(profile="full")

import torch.nn.functional as F

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total
 
# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        # print(type(batch_data))
        # input()
        # batch_data = Variable(batch_data, requires_grad=True)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)
        
        
        # segSize = (batch_data[0]['img_ori'].shape[0],
        #           batch_data[0]['img_ori'].shape[1])
        
        # forward pass
        loss, acc = segmentation_module(batch_data)
        # print(loss)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder,net_decoder1, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    dict_decoder1 = net_decoder1.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))
        
    torch.save(
        dict_decoder1,
        '{}/decoder1_epoch_{}.pth'.format(cfg.DIR, epoch))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder,net_decoder1, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder1 = torch.optim.SGD(
        group_weight(net_decoder1),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder, optimizer_decoder1)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder, optimizer_decoder1) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder
    for param_group in optimizer_decoder1.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    # net_encoder,net_encoder1 = ModelBuilder.build_encoder(
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder,net_decoder1 = ModelBuilder.build_decoder(
    # net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        weights1 = cfg.MODEL.weights_decoder1)

    crit = nn.NLLLoss(ignore_index=-1)
    # crit = FocalLoss(gamma=2,ignore_index=-1)

    # crit2 = nn.MSELoss()
    # crit = nn.NLLLoss(ignore_index=-1)
    
    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder,net_decoder1, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder,net_decoder1, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg)

        # checkpointing
        checkpoint(nets, history, cfg, epoch+1)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder1 = os.path.join(
            cfg.DIR, 'decoder1_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder
    
    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
