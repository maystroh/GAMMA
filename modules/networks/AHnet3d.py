import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.FloatTensor)


def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv3x3x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 1),
        stride=stride,
        padding=(1, 1, 0),
        bias=False)


class BasicBlock3x3x1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3x3x1, self).__init__()
        self.conv1 = conv3x3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x1(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock3x3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3x3x3, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3x3x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 3, 1),
            stride=stride,
            padding=(1, 1, 0),
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if out.size() != residual.size():
                out = self.pool(out)

        out += residual
        out = self.relu(out)

        return out


class Projection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Projection, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False))


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = Pseudo3DLayer(num_input_features + i * growth_rate,
                                  growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class UpTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(UpTransition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False))
        self.add_module('pool', nn.Upsample(scale_factor=2, mode='trilinear'))


class Final(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Final, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=(3, 3, 1),
                stride=1,
                padding=(1, 1, 0),
                bias=False))
        self.add_module('pool', nn.Upsample(scale_factor=2, mode='trilinear'))


class Pseudo3DLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(Pseudo3DLayer, self).__init__()
        # 1x1x1
        self.bn1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False)

        # 3x3x1
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=(3, 3, 1),
            stride=1,
            padding=(1, 1, 0),
            bias=False)

        # 1x1x3
        self.bn3 = nn.BatchNorm3d(growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(
            growth_rate,
            growth_rate,
            kernel_size=(1, 1, 3),
            stride=1,
            padding=(0, 0, 1),
            bias=False)

        # 1x1x1
        self.bn4 = nn.BatchNorm3d(growth_rate)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(
            growth_rate, growth_rate, kernel_size=1, stride=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        inx = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x3x3x1 = self.conv2(x)

        x = self.bn3(x3x3x1)
        x = self.relu3(x)
        x1x1x3 = self.conv3(x)

        x = x3x3x1 + x1x1x3
        x = self.bn4(x)
        x = self.relu4(x)
        new_features = self.conv4(x)

        self.drop_rate = 0  # Dropout will make trouble!
        # since we use the train mode for inference
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([inx, new_features], 1)


class PVP(nn.Module):
    def __init__(self, in_ch):
        super(PVP, self).__init__()
        self.pool64 = nn.MaxPool3d(kernel_size=(64, 64, 1), stride=(64, 64, 1))
        self.pool32 = nn.MaxPool3d(kernel_size=(32, 32, 1), stride=(32, 32, 1))
        self.pool16 = nn.MaxPool3d(kernel_size=(16, 16, 1), stride=(16, 16, 1))
        self.pool8 = nn.MaxPool3d(kernel_size=(8, 8, 1), stride=(8, 8, 1))

        self.proj64 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj32 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj16 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj8 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))

    def forward(self, x):
        x64 = F.upsample(
            self.proj64(self.pool64(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x32 = F.upsample(
            self.proj32(self.pool32(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x16 = F.upsample(
            self.proj16(self.pool16(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x8 = F.upsample(
            self.proj8(self.pool8(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x = torch.cat((x64, x32, x16, x8), dim=1)
        return x


class AHNet(nn.Module):
    def __init__(self, in_channels=1, layers=[3, 4, 6, 3], num_classes=1000):
        self.inplanes = 64
        super(AHNet, self).__init__()
        # Todo: double check why Wenpei added this linear regression layer before the first conv layer..
        self.linear = nn.Linear(2048, 2048)
        # Make the 3x3x1 resnet layers
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(7, 7, 3),
            stride=(2, 2, 1),
            padding=(3, 3, 1),
            bias=False)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.bn0 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(
            Bottleneck3x3x1, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(
            Bottleneck3x3x1, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            Bottleneck3x3x1, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            Bottleneck3x3x1, 512, layers[3], stride=2)

        # Make the 3D dense decoder layers
        DENSEGROWTH = 20
        DENSEBN = 4
        NDENSELAYER = 3

        num_init_features = 64
        NOUTRES1 = 256
        NOUTRES2 = 512
        NOUTRES3 = 1024
        NOUTRES4 = 2048

        self.linear1 = nn.Sequential(nn.Linear(NOUTRES4, NOUTRES2))
        self.linear2 = nn.Sequential(nn.Linear(NOUTRES2, num_classes))

        # Initialise parameters
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(stride, stride, 1),
                    bias=False),
                nn.MaxPool3d(
                    kernel_size=(1, 1, stride), stride=(1, 1, stride)),
                nn.BatchNorm3d(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, (stride, stride, 1), downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)

        # print('after conv1 x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('conv1 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        x = self.pool1(x)

        # print('after pool1 x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('pool1 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        x = self.bn0(x)

        # print('after bn0 x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('bn0 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        x = self.relu(x)

        # print('after relu x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('relu std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        x = self.maxpool(x)

        # print('after maxpool x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('maxpool std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        fm1 = self.layer1(x)

        # print('after layer1 x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('layer1 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        fm2 = self.layer2(fm1)

        # print('after layer2 x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('layer2 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        fm3 = self.layer3(fm2)

        # print('after layer3 x shape is ' + str(x.shape) )
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('layer3 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        fm4 = self.layer4(fm3)

        # print('after layer4 x shape is ' + str(x.shape))
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('layer4 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        fc1 = nn.functional.adaptive_avg_pool3d(fm4, (1, 1, 1))

        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('after adaptive_avg_pool3d x shape is ' + str(x.shape))
        # print('adaptive_avg_pool3d std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        fc2 = self.linear1(fc1.squeeze())

        # print('after linear1 x shape is ' + str(x.shape))
        # print('max input is ' + str((x.max())) + ' min input is ' + str((x.min())))
        # print('linear1 std input is ' + str((x.std())) + ' mean input is ' + str((x.mean())))

        predict = self.linear2(fc2.squeeze())

        # print('after linear2 x shape is ' + str(predict.shape))
        # print('max input is ' + str((predict.max())) + ' min input is ' + str((predict.min())))
        # print('linear2 std input is ' + str((predict.std())) + ' mean input is ' + str((predict.mean())))

        predict = torch.softmax(predict)

        # print('after sigmoid x shape is ' + str(predict.shape))
        # print('max input is ' + str((predict.max())) + ' min input is ' + str((predict.min())))
        # print('sigmoid std input is ' + str((predict.std())) + ' mean input is ' + str((predict.mean())))

        return predict

    def copy_from(self, net):
        # Copy the initial module CONV1 -- Need special care since
        # we only have one input channel in the 3D network
        p2d, p3d = next(net.conv1.parameters()), next(self.conv1.parameters())

        # From 64x3x7x7 -> 64xracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!3x7x7x1 -> 64x1x7x7x3
        p3d.data = p2d.data.unsqueeze(dim=4).permute(0, 4, 2, 3, 1).clone()

        # Copy the initial module BN1
        copy_bn_param(net.bn0, self.bn0)

        # Copy layer1
        layer_2D = []
        layer_3D = []
        for m1 in net.layer1.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer1.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)

        # Copy layer2
        layer_2D = []
        layer_3D = []
        for m1 in net.layer2.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer2.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)

        # Copy layer3
        layer_2D = []
        layer_3D = []
        for m1 in net.layer3.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer3.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)

        # Copy layer4
        layer_2D = []
        layer_3D = []
        for m1 in net.layer4.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer4.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)


def copy_conv_param(module2d, module3d):
    for p2d, p3d in zip(module2d.parameters(), module3d.parameters()):
        p3d.data[:] = p2d.data.unsqueeze(dim=4).clone()[:]


def copy_bn_param(module2d, module3d):
    for p2d, p3d in zip(module2d.parameters(), module3d.parameters()):
        p3d.data[:] = p2d.data[:]  # Two parameter gamma and beta
