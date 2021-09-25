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
import sys
import argparse
import time
import random
import warnings
warnings.filterwarnings('ignore')
import logging
import datetime
from tqdm import tqdm
import numpy as np

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
import paddle.distributed as dist

from dataloader.dataset import ColorDatasetTrain, ColorDatasetVal
from models.model import Color_model
from models.layers import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer
from utils.utils import AverageMeter, adjust_learning_rate
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume

def main():
    parser = argparse.ArgumentParser(description='Colorization!')
    ## Optimizer
    parser.add_argument('--gpu', default='0,1', help='gpu id')
    parser.add_argument('--num_epoch', default=15, type=int, help='training epoch')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--start_step', default=0, type=int, help='start step')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for data loading')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=40, type=int, help='batch size')
    ## Dataset
    parser.add_argument('--size', default=256, type=int, help='image size')
    parser.add_argument('--crop_size', default = 224, type = int, help = 'size for randomly cropping images')
    parser.add_argument('--data_root', type=str, default='/home/ubuntu/lsz/dataset/imagenet/ILSVRC/Resize',
                        help='path to dataset splits data folder')
    parser.add_argument('--dataset', default='imagenet', type=str,)
    ## Checkpoint
    parser.add_argument('--save_step', type = int, default = 1000, help = 'step size for saving trained models')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    ## Utils
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='eval')

    global args
    args = parser.parse_args()
    
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    
    ## Distribution training
    # initialize parallel environment
    # dist.init_parallel_env()
    
    ## fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    ## Logs
    if args.savename=='default':
        args.savename = 'color_%s_batch%d'%(args.dataset,args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))
    
    ## Dataset
    train_transform = transforms.Compose([
                         transforms.RandomCrop(args.crop_size),
                         transforms.RandomHorizontalFlip(),
                        ])
    # val_transform = transforms.Compose([
    #                      transforms.Scale(args.size),
    #                     ])
    val_transform = None
    train_dataset = ColorDatasetTrain(data_dir=args.data_root, split='train', transform=train_transform)
    val_dataset = ColorDatasetVal(data_dir=args.data_root, split='val', transform=val_transform)
    test_dataset = ColorDatasetVal(data_dir=args.data_root, split='test', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                              drop_last=False, num_workers=4)
    
    ## Model
    model = Color_model()
    # model = paddle.DataParallel(model)
    encode_layer = NNEncLayer()
    boost_layer = PriorBoostLayer()
    nongray_mask = NonGrayMaskLayer()

    # print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    # logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))
    
    ## Loss and Optimizer
    criterion = nn.CrossEntropyLoss(reduction='none', axis=1)
    optimizer = optim.Adam(
            parameters= model.parameters(), 
            learning_rate=args.lr, 
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.001)
    if args.pretrain:
        model=load_pretrain(model, args, logging)
    if args.resume:
        model, optimizer = load_resume(model, optimizer, args, logging)
    
    ## Train Val Test
    best_auc = -float('Inf')
    if args.test:
        test_epoch(test_loader, model)
    elif args.eval:
        validate_epoch(val_loader, model)
    else:
        step = args.start_step
        for epoch in range(args.start_epoch, args.num_epoch):
            #--------------------------------------------------------
            batch_time = AverageMeter()
            losses = AverageMeter()
            model.train()
            end = time.time()
            for batch_idx, (images, img_ab) in enumerate(train_loader):
                adjust_learning_rate(optimizer, step)
                images = images.unsqueeze(1)  # [bs, 1, 224, 224]
                images = paddle.cast(images, dtype='float32')
                img_ab = paddle.cast(img_ab, dtype='float32')  # [bs, 2, 56, 56]
                
                ## Preprocess data
                encode, max_encode = encode_layer.forward(img_ab)  # Paper Eq(2) Z空间ground-truth的计算
                targets = paddle.to_tensor(max_encode, dtype = 'int64')
                boost = paddle.to_tensor(boost_layer.forward(encode), dtype='float32')  # Paper Eq(3)-(4), [bs, 1, 56, 56], 每个空间位置的ab概率
                mask = paddle.to_tensor(nongray_mask.forward(img_ab), dtype='float32')  # ab通道数值和小于5的空间位置不计算loss, [bs, 1, 1, 1]
                boost_nongray = boost * mask  # [bs, 1, 56, 56]
                
                ## Model forward
                outputs = model(images)  # [bs, 313, 56, 56]
                
                ## Compute loss
                loss = paddle.mean((criterion(outputs, targets)*(boost_nongray.squeeze(1))))
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), images.shape[0])
            
                ## Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if step % args.print_freq == 0:
                    print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'vis_lr {vis_lr:.8f}\t' \
                        'step {step:.0f}\t' \
                        .format( \
                            epoch, batch_idx, len(train_loader), \
                            loss=losses, vis_lr = optimizer.get_lr(), step = step)
                    print(print_str)
                    logging.info(print_str)
                    
                if step % args.save_step == 0:
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_loss': losses.avg,
                        'optimizer' : optimizer.state_dict(),
                        'step': step+1,
                        }, False, args, filename='colornet_'+str(step))
                step += 1
            #--------------------------------------------------------
            

def validate_epoch(val_loader, model, mode='val'):
    pass
        

def test_epoch(val_loader, model, mode='test'):
    pass


if __name__ == "__main__":
    main()
    # dist.spawn(main)
    # FLAGS_cudnn_deterministic=True python train.py

