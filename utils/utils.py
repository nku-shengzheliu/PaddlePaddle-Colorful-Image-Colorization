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

import numpy as np
from PIL import Image
import numpy as np
from paddle.framework import dtype
from skimage import color

import paddle
import paddle.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def multiclass_metrics(pred, gt):
  """
  check precision and recall for predictions.
  Output: overall = {precision, recall, f1}
  """
  eps=1e-6
  overall = {'precision': -1, 'recall': -1, 'f1': -1}
  NP, NR, NC = 0, 0, 0  # num of pred, num of recall, num of correct
  for ii in range(pred.shape[0]):
    pred_ind = np.array(pred[ii]>0.5, dtype=int)
    gt_ind = np.array(gt[ii]>0.5, dtype=int)
    inter = pred_ind * gt_ind
    # add to overall
    NC += np.sum(inter)
    NP += np.sum(pred_ind)
    NR += np.sum(gt_ind)
  if NP > 0:
    overall['precision'] = float(NC)/NP
  if NR > 0:
    overall['recall'] = float(NC)/NR
  if NP > 0 and NR > 0:
    overall['f1'] = 2*overall['precision']*overall['recall']/(overall['precision']+overall['recall']+eps)
  return overall

def adjust_learning_rate(optimizer, step):
    if step <200000:
        lr = 3*1e-5
    elif step < 375000:
        lr = 1e-5
    else:
        lr = 3*1e-6
    optimizer.set_lr(lr)
    
def load_img(img_path):
  out_np = np.asarray(Image.open(img_path))
  if(out_np.ndim==2):
    out_np = np.tile(out_np[:,:,None],3)
  return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
  # return original size L and resized L as Tensors
  img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
  img_lab_orig = color.rgb2lab(img_rgb_orig)
  img_lab_rs = color.rgb2lab(img_rgb_rs)

  img_l_orig = img_lab_orig[:,:,0]
  img_l_rs = img_lab_rs[:,:,0]

  tens_orig_l = paddle.to_tensor(img_l_orig, dtype='float32')
  tens_orig_l = tens_orig_l.unsqueeze(0).unsqueeze(1)
  tens_rs_l = paddle.to_tensor(img_l_rs, dtype='float32')
  tens_rs_l = tens_rs_l.unsqueeze(0).unsqueeze(1)

  return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = paddle.concat([tens_orig_l, out_ab_orig], axis=1)
	return color.lab2rgb(out_lab_orig.cpu().numpy()[0,...].transpose((1,2,0)))