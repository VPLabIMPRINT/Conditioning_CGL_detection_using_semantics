import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt

import random

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        
        # n = random.randint(0,5)
        # img2 = img.copy()
        # r = img2[0,:,:]
        # g = img2[1,:,:]
        # b = img2[2,:,:]
        # if n == 0:
        #     img2[0,:,:] = r
        #     img2[1,:,:] = b
        #     img2[2,:,:] = g
        # if n == 1:
        #     img2[0,:,:] = g
        #     img2[1,:,:] = b
        #     img2[2,:,:] = r
        # if n == 2:
        #     img2[0,:,:] = g
        #     img2[1,:,:] = r
        #     img2[2,:,:] = b
        # if n == 3:
        #     img2[0,:,:] = b
        #     img2[1,:,:] = r
        #     img2[2,:,:] = g
        # if n == 4:
        #     img2[0,:,:] = b
        #     img2[1,:,:] = g
        #     img2[2,:,:] = r
        img = self.normalize(torch.from_numpy(img.copy()))
        # img = self.normalize(torch.from_numpy(img2.copy()))
        return img
    
    def depth_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img))
        new_img = img.copy()
        img = torch.from_numpy(new_img.copy()).long()
        return img
    
    def depth_input_transform(self, img):
        # 0-255 to 0-1
        # img = 300 * np.float32(np.array(img)) / 255.
        img = np.float32(np.array(img))
        new_img = img.copy()
        img = torch.from_numpy(new_img.copy()).long()
        return img
        
    def segm_transform(self, segm):
        # to tensor, -1 to 149
        # print(np.unique(torch.from_numpy(np.array(segm))))
        segm = torch.from_numpy(np.array(segm)).long() - 1
        # print(np.unique(segm))
        # input()
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            # this_scale = 1
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        # batch_depths_input = torch.zeros(
        #     self.batch_per_gpu,1, batch_height,batch_width)
        #     torch.zeros(
        batch_depths_input = torch.zeros(self.batch_per_gpu,
            batch_height, #// self.segm_downsampling_rate,
            batch_width, #// self.segm_downsampling_rate
            )
        batch_depths = torch.zeros(
            self.batch_per_gpu, batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate
            ).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
            
            img = Image.open(image_path).convert('RGB')
            depth = Image.open(image_path.replace("training","training_processed_depth").replace(".jpg",".png")).convert('L')
            #depth_input = Image.open(image_path.replace("training","training_depth_cgl")).convert('L')
            segm = Image.open(segm_path).convert("L")
            
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')
            depth = imresize(depth, (batch_widths[i], batch_heights[i]), interp='nearest')
            #depth_input = imresize(depth_input, (batch_widths[i], batch_heights[i]), interp='bilinear')
            
            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')
                
            depth_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            depth_rounded.paste(depth, (0, 0))
            depth = imresize(
                depth_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')
                
            # comment this for normal size depth map
            # depth_rounded_inp = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            # depth_rounded_inp.paste(depth_input, (0, 0))
            # depth_input = imresize(
            #     depth_rounded_inp,
            #     (segm_rounded.size[0] // self.segm_downsampling_rate, \
            #      segm_rounded.size[1] // self.segm_downsampling_rate), \
            #     interp='bilinear')

            img = self.img_transform(img)
            
            depth = self.depth_transform(depth)
            #depth_input = self.depth_input_transform(depth_input)
            segm = self.segm_transform(segm)
            
            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            #batch_depths_input[i][:depth_input.shape[0], :depth_input.shape[1]] = depth_input
            batch_depths[i][:depth.shape[0], :depth.shape[1]] = depth
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm
        
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        output['depth'] = batch_depths
        #output['depth_input'] = batch_depths_input
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path).convert('L')
        # try:
        #         # print()
        #         depth = Image.open(image_path.replace("validation","yolo_med_output_png").replace("training","training_processed_depth").replace(".jpg",".png")).convert('L')
        #         # print(np.unique(np.array(depth)))
        # except IOError:
        #         depth = None
            
        #         # print ("ERROR processing ", myimage)
        
    
        # depth = Image.open(image_path.replace("validation","yolo_med_output_png").replace("training","training_processed_depth").replace(".jpg",".png")).convert('L')
        # depth = Image.open(image_path.replace("validation","validation_processed_depth").replace("training","training_processed_depth").replace(".jpg",".png")).convert('L')
        # depth_input = Image.open(image_path.replace("training","training_depth_cgl").replace("validation","validation_depth_cgl")).convert('L')
        
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        # depth_resized_list = []
        # depth_input_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
                
            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')
            # depth_resized = imresize(depth, (target_width, target_height), interp='bilinear')
            # depth_input_res = imresize(depth_input, (target_width, target_height), interp='bilinear')
            
            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

            # if depth is not None:
            # depth_resized = self.depth_transform(depth_resized)
            # depth_resized = torch.unsqueeze(depth_resized, 0)
            # depth_resized_list.append(depth_resized)
            # # else:
            # #     depth_resized_list = []
            # depth_input_res = self.depth_input_transform(depth_input_res)
            # depth_input_res = torch.unsqueeze(depth_input_res, 0)
            # depth_input_list.append(depth_input_res)
            
        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        # output['depth'] = [x.contiguous() for x in depth_resized_list]
        # output['depth_input'] = [x.contiguous() for x in depth_input_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')
        # depth = Image.open(image_path.replace("validation","validation_processed_depth").replace("training","training_processed_depth").replace(".jpg",".png")).convert('L')
        # depth_input = Image.open(image_path.replace("training","training_depth_cgl").replace("validation","validation_depth_cgl")).convert('L')
        ori_width, ori_height = img.size

        img_resized_list = []
        # depth_resized_list = []
        # depth_input_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')
            # depth_resized = imresize(depth, (target_width, target_height), interp='bilinear')
            # depth_input_res = imresize(depth_input, (target_width, target_height), interp='bilinear')
            
            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)
            
            # depth_resized = self.depth_transform(depth_resized)
            # depth_resized = torch.unsqueeze(depth_resized, 0)
            # depth_resized_list.append(depth_resized)
            
            # depth_input_res = self.depth_input_transform(depth_input_res)
            # depth_input_res = torch.unsqueeze(depth_input_res, 0)
            # depth_input_list.append(depth_input_res)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        # output['depth'] = [x.contiguous() for x in depth_resized_list]
        # output['depth_input'] = [x.contiguous() for x in depth_input_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
