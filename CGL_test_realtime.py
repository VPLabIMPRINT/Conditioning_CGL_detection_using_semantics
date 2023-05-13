# System libs
from ast import arg
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
from torchvision import transforms
# Our libs
from dataset_original import TestDataset
from mit_semseg.models.new_models_dflb_trans_var import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg
import time 
import cv2

colors = loadmat('data/color_cgl_realtime.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(data, pred, cfg):
    (img) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    # print("Predictions in [{}]:".format(info))
    # for idx in np.argsort(counts)[::-1]:
    #     name = names[uniques[idx] + 1]
    #     ratio = counts[idx] / pixs * 100
    #     if ratio > 0.1:
    #         print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    pred_color = np.where(pred_color == 120, 0, pred_color)
    im_vis = cv2.addWeighted(img[...,::-1].copy(), 1, pred_color, 1, 0)
    lf2 = cv2.imread("white.png")
    lf2 = cv2.resize(lf2, (10, pred_color.shape[0]))
    
    # cv2.imwrite('white.png',np.full((50, width,3), 255))
    img = cv2.hconcat([img, lf2])
    im_vis = np.concatenate((img[...,::-1].copy(), im_vis), axis=1)
    # milliseconds = int(round(time.time() * 1000))
    # print(milliseconds)
    cv2.namedWindow("Covert Geo-Location (CGL) Detection Demo", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Covert Geo-Location (CGL) Detection Demo",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Covert Geo-Location (CGL) Detection Demo',im_vis[...,::-1].copy())
    cv2.waitKey(100)

def test(segmentation_module, gpu, batch_data):
    segmentation_module.eval()
    
    # batch_data = batch_data[0]
    segSize = (batch_data['img_ori'].shape[0],
                batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    with torch.no_grad():
        scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
        scores = async_copy_to(scores, gpu)

        for img in img_resized_list:
            feed_dict = batch_data.copy()
            feed_dict['img_data'] = img
            del feed_dict['img_ori']
            feed_dict = async_copy_to(feed_dict, gpu)

            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

    # visualization
    visualize_result(
        (batch_data['img_ori']),
        pred,
        cfg
    )

normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img
    
def main(cfg, gpu):
    # Comment this line for execution on cpu
    torch.cuda.set_device(gpu)

    # Uncomment the following two lines for execution on cpu
    # device = torch.device("cpu")
    # gpu = device
    
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder, net_decoder1 = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        weights1=cfg.MODEL.weights_decoder1,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder,net_decoder1, crit)
    
    # comment the following line to execute on cpu
    segmentation_module.cuda()
    vid = cv2.VideoCapture(0)
    # for i in range(2000):
    
    while True:
        ret, frame = vid.read()
        img_list = []
        img = img_transform(frame[...,::-1].copy())
        img = torch.unsqueeze(img, 0)
        img_list.append(img)

        output = dict()
        output['img_ori'] = np.array(frame)
        output['img_data'] = [x.contiguous() for x in img_list]
        
        test(segmentation_module, gpu, output)

        
    # Dataset and Loader
    # dataset_test = TestDataset(
    #     cfg.list_test,
    #     cfg.DATASET)
    # loader_test = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=cfg.TEST.batch_size,
    #     shuffle=False,
    #     collate_fn=user_scattered_collate,
    #     num_workers=5,
    #     drop_last=True)

    
    # test(segmentation_module, loader_test, gpu, output)
    
if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    # parser.add_argument(
    #     "--imgs",
    #     required=True,
    #     type=str,
    #     help="an image paths, or a directory name"
    # )
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
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="a directory name"
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

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder1 = os.path.join(
        cfg.DIR, 'decoder1_' + cfg.TEST.checkpoint)

    print(cfg.MODEL.weights_encoder)
    print(cfg.MODEL.weights_decoder)
    print(cfg.MODEL.weights_decoder1)
    
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    
    # generate testing image list
    # if os.path.isdir(args.imgs[0]):
        
    #     imgs = os.listdir(args.imgs)
    #     n = []
    #     for filename in imgs:
    #         n.append(args.imgs + "/" + filename)
    #     imgs = n
    #     # imgs = find_recursive(args.imgs[0])
    # else:
    #     imgs = [args.imgs]
    # assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    # cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
