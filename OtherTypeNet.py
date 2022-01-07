import torchvision.models as models
import torch.nn as nn
import torch
from thop import profile

df = 15


def VGG16(df, num_class=1,):
    net = models.vgg16()
    net.classifier = nn.Sequential(
        nn.Linear(7*7*512, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Mobilenetv2(df, num_class=1,):
    net = models.mobilenet_v2()
    net.classifier = nn.Sequential(
        nn.Linear(1280, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Densenet121(df, num_class=1,):
    net = models.densenet121()
    net.classifier = nn.Sequential(
        nn.Linear(1024, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Resnext50(df, num_class=1,):
    net = models.resnext50_32x4d()
    net.classifier = nn.Sequential(
        nn.Linear(2048, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def AlexNet(df, num_class=1,):
    net = models.AlexNet()
    net.classifier = nn.Sequential(
        nn.Linear(9216, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Mnasnet(df, num_class=1,):
    net = models.mnasnet1_0()
    net.classifier = nn.Sequential(
        nn.Linear(1280, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Resnet101(df, num_class=1,):
    net = models.resnet101()
    net.fc = nn.Sequential(
        nn.Linear(2048, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net


def Resnet50(df, num_class=1,):
    net = models.resnet50()
    net.fc = nn.Sequential(
        nn.Linear(2048, df),
        nn.ReLU(inplace=True),
        nn.Linear(df, num_class)
    )
    return net

# # test
# print(resnet50(df=df))
# net = vgg16(df)
# input = torch.randn(1, 3, 224, 224)
# out = net(input)
# print("构建网络成功")

# 计算 flops 和 parmas
# print(VGG16(df=df))
# print(Mobilenetv2(df=df))
# print(Densenet121(df=df))
# print(Resnext50(df=df))
# print(AlexNet(df=df))
# print(Mnasnet(df=df))
# print(Resnet50(df=df))
# Nets = [Densenet121(df), AlexNet(df), Mobilenetv2(df), VGG16(df), Mnasnet(df), Resnext50(df), Resnet50(df)]
# Nets_names = ['Densenet121()', 'AlexNet()', 'Mobilenetv2()', 'VGG16()', 'Mnasnet()', 'Resnext50()', 'Resnet50()']
# input = torch.randn(1, 3, 224, 224)
# for Net, Name in zip(Nets, Nets_names):
#     num_params = 0
#     for param in Net.parameters():
#         num_params += param.numel()
#     # print(Name, num_params / 1e6)
#     flops, params = profile(Net, inputs=(input,))
#     print(Name, params / 1e6, flops / 1e9)

