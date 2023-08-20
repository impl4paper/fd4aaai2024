import torch
import torch.nn as nn
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}





class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, mesh_flag=False):
        super(VGG, self).__init__()
        self.mesh_flag = mesh_flag
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.mesh_flag:
            mesh_vector = out.clone().detach()
            out = self.classifier(out)
            return mech_vector, out
        else:
            out = self.classifier(out)
            return out


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class VGGclassifier(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGGclassifier, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                in_channels = x
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def get_vgg16(num_classes=10):
    return VGG('VGG16', num_classes)

def get_vgg11(num_classes=10):
    return VGG('VGG11', num_classes)

def get_vgg19(num_classes=10):
    return VGG('VGG19', num_classes)
def get_vgg13(num_classes=10):
    return VGG('VGG13', num_classes)


def get_split_vgg19(num_classes=10):

    node_cfg_0 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M']
    node_cfg_1 = [ 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
   
    layer0 = VggLayer(node_cfg_0)
    layer1 = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2], last_flag=True, num_classes=num_classes)
    return layer0, layer1

if __name__ == '__main__':
   
    node_cfg_0 = [64, 64, 'M', 128, 128, 'M']
    node_cfg_1 = [256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layer0 = VggLayer(node_cfg_0)
    layer1 = VggLayer(node_cfg_1, node_cfg_0[-1] if node_cfg_0[-1] != 'M' else node_cfg_0[-2], last_flag=True)
    










