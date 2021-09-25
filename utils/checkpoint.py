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
import shutil
import numpy as np

import paddle

def save_checkpoint(state, is_best, args, filename='default'):
    if filename=='default':
        filename = 'color_%s_batch%d'%(args.dataset,args.batch_size)

    checkpoint_name = './saved_models/%s_checkpoint.pdparams'%(filename)
    best_name = './saved_models/%s_model_best.pdparams'%(filename)
    paddle.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)

def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        checkpoint = paddle.load(args.pretrain)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        model.load_dict(model_dict)
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        logging.info("=> loaded pretrain model at {}"
              .format(args.pretrain))
        del checkpoint  # dereference seems crucial
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
        logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    return model

def load_resume(model, optimizer, args, logging):
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = paddle.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.start_step = checkpoint['step']
        best_loss = checkpoint['best_loss']
        model.load_dict(checkpoint['state_dict'])
        optimizer.set_state_dict(checkpoint['optimizer'])
        print(("=> loaded checkpoint (epoch {}) Loss{}"
              .format(checkpoint['epoch'], best_loss)))
        logging.info("=> loaded checkpoint (epoch {}) Loss{}"
              .format(checkpoint['epoch'], best_loss))
        del checkpoint  # dereference seems crucial
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        logging.info(("=> no checkpoint found at '{}'".format(args.resume)))
    return model, optimizer

if __name__=="__main__":
    path = 'saved_models/colornet_0_checkpoint.pdparams'
    checkpoint = paddle.load(path)
    from models.model import Color_model
    import paddle.optimizer as optim
    model = Color_model()
    optimizer = optim.Adam(
            parameters= model.parameters(), 
            learning_rate=0.001, 
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.001)
    model.load_dict(checkpoint['state_dict'])
    optimizer.set_state_dict(checkpoint['optimizer'])
    print("load success!!!")