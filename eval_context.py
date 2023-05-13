# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset_original import ValDataset
from mit_semseg.models.context_models3 import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']

def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    # Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()
    # print(segmentation_module.encoder.features[11].context_layer.weight)
    # input()
    # print(segmentation_module.encoder1.add_layer.context_layer.weight)
    # input()
    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        # depth_resized_list = batch_data['depth']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)
            
            count = 0
            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)
                count += 1    
                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder, net_encoder1 = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
	weights1=cfg.MODEL.weights_encoder1)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)
    crit1 = nn.MSELoss()
    segmentation_module = SegmentationModule(net_encoder,net_encoder1, net_decoder, crit, crit1)
    
    # h = 0.04167 # 1 / 24
    
    # segmentation_module.encoder.features[11].context_layer.weight = nn.Parameter(torch.tensor([[h,h,h,h,h],[h,h,h,h,h],[h,h,0,h,h],[h,h,h,h,h],[h,h,h,h,h]]).repeat(segmentation_module.encoder.features[11].context_layer.in_channels, segmentation_module.encoder.features[11].context_layer.out_channels,1,1).cuda())
    # segmentation_module.encoder.features[19].context_layer.weight = nn.Parameter(torch.tensor([[h,h,h,h,h],[h,h,h,h,h],[h,h,0,h,h],[h,h,h,h,h],[h,h,h,h,h]]).repeat(segmentation_module.encoder.features[19].context_layer.in_channels, segmentation_module.encoder.features[19].context_layer.out_channels,1,1).cuda())
    # segmentation_module.encoder.features[11].context_layer.eval()
    # segmentation_module.encoder.features[19].context_layer.eval()
    # segmentation_module.encoder1.add_layer.context_layer.weight = nn.Parameter(torch.tensor([[h,h,h,h,h],[h,h,h,h,h],[h,h,0,h,h],[h,h,h,h,h],[h,h,h,h,h]]).repeat(segmentation_module.encoder1.add_layer.context_layer.in_channels, segmentation_module.encoder1.add_layer.context_layer.out_channels,1,1).cuda()) 
    # segmentation_module.encoder1.add_layer.context_layer.eval()

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
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

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_encoder1 = os.path.join(
        cfg.DIR, 'encoder1_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)
