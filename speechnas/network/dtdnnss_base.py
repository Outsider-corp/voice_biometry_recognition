from __future__ import print_function, division
import os

os.chdir(r'D:\Py_Projects\neuro')
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechnas.nas_core.global_enum import Activation
from speechnas.nas_core.layer.common_layer.conv1d import Conv1d_
from speechnas.nas_core.layer.common_layer.dtdnn import TransitLayer, DenseLayer
from speechnas.nas_core.layer.common_layer.dtdnnss import DTDNNSS_
from speechnas.nas_core.layer.common_layer.tdnn import TDNN_, StatisticsPooling

LEN_FRAME = 200


def for_extract_embedding(maxChunk=10000, isMatrix=True):
    """
    A decorator for extract_embedding class-function to wrap some common process codes like Kaldi's x-vector extractor.
    Used in TopVirtualNnet.
    """

    def wrapper(function):
        def _wrapper(self, input):
            """
            @input: a 3-dimensional tensor with batch-dim=1 or [frames, feature-dim] matrix for
                    acoustic features only
            @return: an 1-dimensional vector
            """
            train_status = self.training
            self.eval()

            with torch.no_grad():
                if isMatrix:
                    input = torch.tensor(input)
                    input = torch.unsqueeze(input, dim=0)
                    input = input.transpose(1, 2)

                input = input.cuda()
                num_frames = input.shape[2]
                num_split = (num_frames + maxChunk - 1) // maxChunk
                split_size = num_frames // num_split

                offset = 0
                embedding_stats = 0.
                for i in range(0, num_split - 1):
                    this_embedding = function(self, input[:, :, offset:offset + split_size])
                    offset += split_size
                    embedding_stats += split_size * this_embedding

                last_embedding = function(self, input[:, :, offset:])

                embedding = (embedding_stats + (num_frames - offset) * last_embedding) / num_frames

                if train_status:
                    self.train()

                return torch.squeeze(embedding.transpose(1, 2)).cpu()

        return _wrapper

    return wrapper


class DtdnnssBase(nn.Module):
    def __init__(self, num_class, in_channels: int = 20, mid_channels=128, feature_dim=512, loss_type='aam_loss'):
        super(DtdnnssBase, self).__init__()
        self.loss_type = loss_type

        # TODO Layer 1
        self.layer1 = TDNN_(in_channels=in_channels, out_channels=128, kernel_size=5, bias=True,
                            dilation=1, in_shape=LEN_FRAME, out_shape=LEN_FRAME,
                            act=Activation.PReLU)

        # TODO Layer 2
        self.layer2 = DTDNNSS_(in_channels=128, mid_channels=mid_channels, out_channels=64,
                               kernel_size=3,
                               dilation=(1, 3), bias=False, in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 3
        self.layer3 = DTDNNSS_(in_channels=192, mid_channels=mid_channels, out_channels=64,
                               kernel_size=3,
                               dilation=(1, 3), bias=False, in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 4
        self.layer4 = DTDNNSS_(in_channels=256, mid_channels=mid_channels, out_channels=64,
                               kernel_size=3,
                               dilation=(1, 3), bias=False, in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 5
        self.layer5 = DTDNNSS_(in_channels=320, mid_channels=mid_channels, out_channels=64,
                               kernel_size=3,
                               dilation=(1, 3), bias=False, in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 6
        self.layer6 = DTDNNSS_(in_channels=384, mid_channels=mid_channels, out_channels=64,
                               kernel_size=3,
                               dilation=(1, 3), bias=False, in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 7
        self.layer7 = DTDNNSS_(in_channels=448, mid_channels=mid_channels, out_channels=64,
                               kernel_size=3,
                               dilation=(1, 3), bias=False, in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 8
        self.layer8 = TransitLayer(in_channels=512, out_channels=256, act=Activation.PReLU,
                                   in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 9
        self.layer9 = DTDNNSS_(in_channels=256, mid_channels=mid_channels, out_channels=64,
                               kernel_size=3,
                               dilation=(1, 3), bias=False, in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO Layer 10
        self.layer10 = DTDNNSS_(in_channels=320, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 11
        self.layer11 = DTDNNSS_(in_channels=384, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 12
        self.layer12 = DTDNNSS_(in_channels=448, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 13
        self.layer13 = DTDNNSS_(in_channels=512, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 14
        self.layer14 = DTDNNSS_(in_channels=576, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 15
        self.layer15 = DTDNNSS_(in_channels=640, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 16
        self.layer16 = DTDNNSS_(in_channels=704, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 17
        self.layer17 = DTDNNSS_(in_channels=768, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 18
        self.layer18 = DTDNNSS_(in_channels=832, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 19
        self.layer19 = DTDNNSS_(in_channels=896, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 20
        self.layer20 = DTDNNSS_(in_channels=960, mid_channels=mid_channels, out_channels=64,
                                kernel_size=3,
                                dilation=(1, 3), bias=False, in_shape=LEN_FRAME,
                                out_shape=LEN_FRAME)

        # TODO Layer 21
        self.layer21 = TransitLayer(in_channels=1024, out_channels=512, act=Activation.PReLU,
                                    in_shape=LEN_FRAME, out_shape=LEN_FRAME)

        # TODO pooling 22
        self.pooling22 = StatisticsPooling(512)

        # TODO Layer 23
        self.layer23 = DenseLayer(in_channels=512 * 2, out_channels=feature_dim, in_shape=1,
                                  out_shape=1)

        # TODO Layer logits
        if self.loss_type == 'aam_loss':
            self.weight = torch.nn.Parameter(torch.randn(num_class, feature_dim, 1),
                                             requires_grad=True)
        else:
            self.logits = Conv1d_(feature_dim, num_class, 1, act=Activation.Identity, in_shape=1,
                                  out_shape=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
                # m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.loss_type == 'aam_loss':
            torch.nn.init.normal_(self.weight, 0., 0.01)  # It seems better.

    def forward(self, inputs, **kwargs):
        # TODO Layer 1
        feature1 = self.layer1(inputs)
        # TODO Layer 2
        feature2 = self.layer2(feature1)
        # TODO Layer 3
        feature3 = self.layer3(feature2)
        # TODO Layer 4
        feature4 = self.layer4(feature3)
        # TODO Layer 5
        feature5 = self.layer5(feature4)
        # TODO Layer 6
        feature6 = self.layer6(feature5)
        # TODO Layer 7
        feature7 = self.layer7(feature6)
        # TODO Layer 8
        feature8 = self.layer8(feature7)
        # TODO Layer 9
        feature9 = self.layer9(feature8)
        # TODO Layer 10
        feature10 = self.layer10(feature9)
        # TODO Layer 11
        feature11 = self.layer11(feature10)
        # TODO Layer 12
        feature12 = self.layer12(feature11)
        # TODO Layer 13
        feature13 = self.layer13(feature12)
        # TODO Layer 14
        feature14 = self.layer14(feature13)
        # TODO Layer 15
        feature15 = self.layer15(feature14)
        # TODO Layer 16
        feature16 = self.layer16(feature15)
        # TODO Layer 17
        feature17 = self.layer17(feature16)
        # TODO Layer 18
        feature18 = self.layer18(feature17)
        # TODO Layer 19
        feature19 = self.layer19(feature18)
        # TODO Layer 20
        feature20 = self.layer20(feature19)
        # TODO Layer 21
        feature21 = self.layer21(feature20)
        # TODO Layer 22
        feature22 = self.pooling22(feature21)
        # TODO Layer 23
        feature23 = self.layer23(feature22)

        # get madds
        self.madds = 0
        self.param = 0
        self.madds += self.layer1.multiply_adds
        self.madds += self.layer2.multiply_adds
        self.madds += self.layer3.multiply_adds
        self.madds += self.layer4.multiply_adds
        self.madds += self.layer5.multiply_adds
        self.madds += self.layer6.multiply_adds
        self.madds += self.layer7.multiply_adds
        self.madds += self.layer8.multiply_adds
        self.madds += self.layer9.multiply_adds
        self.madds += self.layer10.multiply_adds
        self.madds += self.layer11.multiply_adds
        self.madds += self.layer12.multiply_adds
        self.madds += self.layer13.multiply_adds
        self.madds += self.layer14.multiply_adds
        self.madds += self.layer15.multiply_adds
        self.madds += self.layer16.multiply_adds
        self.madds += self.layer17.multiply_adds
        self.madds += self.layer18.multiply_adds
        self.madds += self.layer19.multiply_adds
        self.madds += self.layer20.multiply_adds
        self.madds += self.layer21.multiply_adds
        # self.madds += self.pooling22.multiply_adds
        self.madds += self.layer23.multiply_adds

        # get params
        self.param += self.layer1.params
        self.param += self.layer2.params
        self.param += self.layer3.params
        self.param += self.layer4.params
        self.param += self.layer5.params
        self.param += self.layer6.params
        self.param += self.layer7.params
        self.param += self.layer8.params
        self.param += self.layer9.params
        self.param += self.layer10.params
        self.param += self.layer11.params
        self.param += self.layer12.params
        self.param += self.layer13.params
        self.param += self.layer14.params
        self.param += self.layer15.params
        self.param += self.layer16.params
        self.param += self.layer17.params
        self.param += self.layer18.params
        self.param += self.layer19.params
        self.param += self.layer20.params
        self.param += self.layer21.params
        # self.param += self.pooling22.params
        self.param += self.layer23.params

        # TODO logits
        if self.loss_type == 'aam_loss':
            normalized_x = F.normalize(feature23.squeeze(dim=2), dim=1)
            normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
            cosine_theta = F.linear(normalized_x, normalized_weight)  # Y = W*X
            return cosine_theta
        else:
            result = self.logits(feature23)
            self.madds += self.logits.multiply_adds
            self.param += self.logits.params
            return result

    def get_madds(self):
        return self.madds

    def get_param(self):
        return self.param

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        # TODO Layer 1
        feature1 = self.layer1(inputs)
        # TODO Layer 2
        feature2 = self.layer2(feature1)
        # TODO Layer 3
        feature3 = self.layer3(feature2)
        # TODO Layer 4
        feature4 = self.layer4(feature3)
        # TODO Layer 5
        feature5 = self.layer5(feature4)
        # TODO Layer 6
        feature6 = self.layer6(feature5)
        # TODO Layer 7
        feature7 = self.layer7(feature6)
        # TODO Layer 8
        feature8 = self.layer8(feature7)
        # TODO Layer 9
        feature9 = self.layer9(feature8)
        # TODO Layer 10
        feature10 = self.layer10(feature9)
        # TODO Layer 11
        feature11 = self.layer11(feature10)
        # TODO Layer 12
        feature12 = self.layer12(feature11)
        # TODO Layer 13
        feature13 = self.layer13(feature12)
        # TODO Layer 14
        feature14 = self.layer14(feature13)
        # TODO Layer 15
        feature15 = self.layer15(feature14)
        # TODO Layer 16
        feature16 = self.layer16(feature15)
        # TODO Layer 17
        feature17 = self.layer17(feature16)
        # TODO Layer 18
        feature18 = self.layer18(feature17)
        # TODO Layer 19
        feature19 = self.layer19(feature18)
        # TODO Layer 20
        feature20 = self.layer20(feature19)
        # TODO Layer 21
        feature21 = self.layer21(feature20)
        # TODO Layer 22
        feature22 = self.pooling22(feature21)
        # TODO Layer 23
        feature23 = self.layer23(feature22)

        # get madds
        self.madds = 0
        self.param = 0
        self.madds += self.layer1.multiply_adds
        self.madds += self.layer2.multiply_adds
        self.madds += self.layer3.multiply_adds
        self.madds += self.layer4.multiply_adds
        self.madds += self.layer5.multiply_adds
        self.madds += self.layer6.multiply_adds
        self.madds += self.layer7.multiply_adds
        self.madds += self.layer8.multiply_adds
        self.madds += self.layer9.multiply_adds
        self.madds += self.layer10.multiply_adds
        self.madds += self.layer11.multiply_adds
        self.madds += self.layer12.multiply_adds
        self.madds += self.layer13.multiply_adds
        self.madds += self.layer14.multiply_adds
        self.madds += self.layer15.multiply_adds
        self.madds += self.layer16.multiply_adds
        self.madds += self.layer17.multiply_adds
        self.madds += self.layer18.multiply_adds
        self.madds += self.layer19.multiply_adds
        self.madds += self.layer20.multiply_adds
        self.madds += self.layer21.multiply_adds
        # self.madds += self.pooling22.multiply_adds
        self.madds += self.layer23.multiply_adds

        # get params
        self.param += self.layer1.params
        self.param += self.layer2.params
        self.param += self.layer3.params
        self.param += self.layer4.params
        self.param += self.layer5.params
        self.param += self.layer6.params
        self.param += self.layer7.params
        self.param += self.layer8.params
        self.param += self.layer9.params
        self.param += self.layer10.params
        self.param += self.layer11.params
        self.param += self.layer12.params
        self.param += self.layer13.params
        self.param += self.layer14.params
        self.param += self.layer15.params
        self.param += self.layer16.params
        self.param += self.layer17.params
        self.param += self.layer18.params
        self.param += self.layer19.params
        self.param += self.layer20.params
        self.param += self.layer21.params
        # self.param += self.pooling22.params
        self.param += self.layer23.params

        return feature23


if __name__ == "__main__":
    from torchsummary import summary

    model = DtdnnssBase(num_class=7232, feature_dim=128).to('cuda')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (30, 200))
