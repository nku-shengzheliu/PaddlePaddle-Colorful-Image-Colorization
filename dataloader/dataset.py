#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import copy
import numpy as np
from PIL import Image
import cv2 as cv
from skimage.color import rgb2lab, rgb2gray

import paddle
import paddle.io as io
import paddle.vision.transforms as transforms


class ColorDatasetTrain(io.Dataset):
    def __init__(self, data_dir, split, transform):
        self.data_dir=os.path.join(data_dir, split)
        self.file_list=os.listdir(self.data_dir)
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(osp.join(self.data_dir, self.file_list[index]))
        # img = cv.imread(osp.join(self.data_dir, self.file_list[index]))
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.transform is not None:
            img_original = self.transform(img)
        else:
            img_original = copy.deepcopy(img)
        img_resize = transforms.Resize(56)(img_original)  # 将224*224的输入图像缩放到56*56，对应model最终的输出[56*56*313]
        img_original = np.asarray(img_original)
        img_lab = rgb2lab(img_resize)
        img_ab = img_lab[:, :, 1:3]
        img_ab = paddle.to_tensor(img_ab.transpose((2, 0, 1)))  # [ab, H/4, W/4]即[2,56,56]
        img_original = rgb2lab(img_original)[:,:,0]-50.  # 取明度通道， subtract 50 for mean-centering
        img_original = paddle.to_tensor(img_original)  # [224,224,1]
        return img_original, img_ab
    def __len__(self):
        return len(self.file_list)


class ColorDatasetVal(io.Dataset):
    def __init__(self, data_dir, split, transform):
        self.data_dir=os.path.join(data_dir, split)
        if split == 'val':
            # TODO: 使用imagenet-val前10000 images验证
            self.file_list = os.listdir(self.data_dir)
        elif split == 'test':
            # 使用论文"Learning Representations for Automatic Colorization"
            # 提出的10000张来自imagenet-val的测试图像
            with open('dataloader/ctest.txt', 'r') as f:
                file_list = f.readlines()
            self.file_list = []
            for file in file_list:
                file = file.strip().split(' ')[0]
                file = file.split('/')[-1]
                self.file_list.append(file)
        self.transform = transform

    def __getitem__(self, index):
        img=Image.open(osp.join(self.data_dir, self.file_list[index]))
        # img = cv.imread(osp.join(self.data_dir, self.file_list[index]))
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.transform != None:
            img_scale = self.transform(img)
        img_scale = np.asarray(img_scale)
        # img_scale = rgb2gray(img_scale)
        img_scale = rgb2lab(img_scale)[:,:,0]  # 取明度通道
        img_scale_process = img_scale - 50.  #  subtract 50 for mean-centering
        img_scale_process = paddle.to_tensor(img_scale_process)  # [224,224,1]
        img_scale = paddle.to_tensor(img_scale)  # [224,224,1]
        return img_scale, img_scale_process

    def __len__(self):
        return len(self.file_list)
