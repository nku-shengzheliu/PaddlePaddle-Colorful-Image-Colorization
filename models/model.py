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

import paddle
import paddle.nn as nn

class Color_model(nn.Layer):
    def __init__(self, norm_layer=nn.BatchNorm2D):
        super(Color_model, self).__init__()
        nn.initializer.set_global_initializer(nn.initializer.XavierNormal(), nn.initializer.Constant())
        model1=[nn.Conv2D(1, 64, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2D(64, 64, kernel_size=3, stride=2, padding=1, bias_attr=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64, momentum=0.1),]

        model2=[nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2D(128, 128, kernel_size=3, stride=2, padding=1, bias_attr=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128, momentum=0.1),]

        model3=[nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2D(256, 256, kernel_size=3, stride=2, padding=1, bias_attr=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256, momentum=0.1),]

        model4=[nn.Conv2D(256, 512, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512, momentum=0.1),]

        model5=[nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias_attr=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias_attr=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias_attr=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512, momentum=0.1),]

        model6=[nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias_attr=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias_attr=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2D(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias_attr=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512, momentum=0.1),]

        model7=[nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512, momentum=0.1),]

        model8=[nn.Conv2DTranspose(512, 256, kernel_size=4, stride=2, padding=1, bias_attr=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias_attr=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2D(256, 313, kernel_size=1, stride=1, padding=0, bias_attr=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

    def forward(self, gray_image):
        conv1_2 = self.model1(gray_image)  # [bs, 64, 112, 112]
        conv2_2 = self.model2(conv1_2)  # [bs, 128, 56, 56]
        conv3_3 = self.model3(conv2_2)  # [bs, 256, 28, 28]
        conv4_3 = self.model4(conv3_3)  # [bs, 512, 28, 28]
        conv5_3 = self.model5(conv4_3)  # [bs, 512, 28, 28]
        conv6_3 = self.model6(conv5_3)  # [bs, 512, 28, 28]
        conv7_3 = self.model7(conv6_3)  # [bs, 512, 28, 28]
        features = self.model8(conv7_3)  # [bs, 313, 56, 56]
        features = features/0.38  # 0.38 is Softmax temperature T. Paper Eq.(5)
        return features

if __name__=="__main__":
    x = paddle.randn([40, 1, 224, 224])*100
    model = Color_model()
    output = model(x)
    print(output.shape)