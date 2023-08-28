import torch
import torch.nn as nn
import torch.nn.functional as F

class basisblock(nn.Module):
    def __init__(self, inplanes, planes, groups = 1):
        super(basisblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.resid = None
        if inplanes != planes:
            self.resid = nn.Conv2d(inplanes, planes, 1, 1, 0, bias = False)

    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        return x

class bottleneck(nn.Module):
    def __init__(self, inplanes, planes, groups = 1):
        super(bottleneck, self).__init__()
        self.resid = None
        if inplanes != planes:
            self.resid = nn.Conv2d(inplanes, planes, 1, 1, 0, bias = False)
            hidplanes = inplanes
        else:
            hidplanes = inplanes // 2
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, hidplanes, 1, 1, 0, bias = False)
        self.bn1 = nn.BatchNorm2d(hidplanes)

        self.conv2 = nn.Conv2d(hidplanes, hidplanes, 3, 1, 1, groups = groups, bias = False)
        self.bn2 = nn.BatchNorm2d(hidplanes)

        self.conv3 = nn.Conv2d(hidplanes, planes, 1, 1, 0, bias = False)
        self.bn3 = nn.BatchNorm2d(planes)
        

    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)

        return x

def switchLayer(channels, xs):
    numofeature = len(xs)
    splitxs = []
    for i in range(numofeature):
        splitxs.append(
            list(torch.chunk(xs[i], numofeature, dim = 1))
        )
    
    for i in range(numofeature):
        h,w = splitxs[i][i].shape[2:]
        tmp = []
        for j in range(numofeature):
            if i > j:
                splitxs[j][i] = F.avg_pool2d(splitxs[j][i], kernel_size = (2**(i-j)))
            elif i < j: 
                # splitxs[j][i] = F.interpolate(splitxs[j][i], (h,w), mode = 'bilinear')
                splitxs[j][i] = F.interpolate(splitxs[j][i], (h,w))
            tmp.append(splitxs[j][i])
        xs[i] = torch.cat(tmp, dim = 1)

    return xs

class FeatureShuffleNet(nn.Module):
    def __init__(self, block, channels = 64, numofblocks = None, groups = 1):
        super(FeatureShuffleNet, self).__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, 2, 3, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        Layerplanes = [self.channels, self.channels, self.channels*2, self.channels*3, self.channels*4]
        self.downSteps = nn.ModuleList()
        for planes in Layerplanes[:-1]:
            self.downSteps.append(
                nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 2, 1, bias = False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(True),
                )
            )

        self.blocks_1 = nn.ModuleList()
        self.blocks_2 = nn.ModuleList()
        self.blocks_3 = nn.ModuleList()


        for l in range(4):
            for i, num in enumerate(self.numofblocks[l]):
                tmp = [block(Layerplanes[i+l], Layerplanes[i+1+l], groups = groups)]
                for j in range(num-1):
                    tmp.append(block(Layerplanes[i+1+l], Layerplanes[i+1+l], groups = groups))
                
                if l == 0:
                    self.blocks_1.append(nn.Sequential(*tmp))
                elif l == 1:
                    self.blocks_2.append(nn.Sequential(*tmp))
                elif l == 2:
                    self.blocks_3.append(nn.Sequential(*tmp))
                else:
                    self.blocks_4 = nn.Sequential(*tmp) # last layer only have 1 block

    def forward(self, x):
        x = self.stem(x) 
        x1 = self.downSteps[0](x) # 64 > H/4, W/4

        x1 = self.blocks_1[0](x1)
        x2 = self.downSteps[1](x1)

        x1 = self.blocks_1[1](x1)
        x2 = self.blocks_2[0](x2)
        x3 = self.downSteps[2](x2)
        x1,x2 = switchLayer(self.channels, [x1,x2])

        x1 = self.blocks_1[2](x1)
        x2 = self.blocks_2[1](x2)
        x3 = self.blocks_3[0](x3)
        x4 = self.downSteps[3](x3)
        x1,x2,x3 = switchLayer(self.channels, [x1,x2,x3])

        x1 = self.blocks_1[3](x1)
        x2 = self.blocks_2[2](x2)
        x3 = self.blocks_3[1](x3)
        x4 = self.blocks_4(x4)

        return x1,x2,x3,x4

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def FSNet_Splus(pretrained = True):
    numofblocks = [
        [4,2,2,2],
        [2,2,2],
        [8,8],
        [4]
    ]
    model = FeatureShuffleNet(basisblock, channels = 64, numofblocks = numofblocks)
    print("FSNet_M parameter size: ", count_parameters(model))
    if pretrained:
        # load_path = "./pretrained/FSNet_M_SynthMLT.pth"
        # cpt = torch.load(load_path)
        # model.load_state_dict(cpt, strict=True)
        # print("load pretrain weight from {}. ".format(load_path))
        print("FSNet_M does not have pretrained weight yet. ")
    return model

def FSNet_M(pretrained = True):
    numofblocks = [
        [4,2,2,2],
        [4,4,4],
        [10,10],
        [10]
    ]
    # model = FeatureShuffleNet(basisblock, channels = 64, numofblocks = numofblocks)
    model = FeatureShuffleNet(bottleneck, channels = 64, numofblocks = numofblocks)
    print("FSNet_M now with bottleneck.")
    print("FSNet_M parameter size: ", count_parameters(model))
    if pretrained:
        # load_path = "./pretrained/FSNet_M_ALL.pth"
        # cpt = torch.load(load_path)
        # model.load_state_dict(cpt, strict=True)
        # print("load pretrain weight from {}. ".format(load_path))
        print("FSNet_M does not have pretrained weight yet. ")
    return model

def FSNeXt_M(pretrained = True):
    numofblocks = [
        [4,2,2,2],
        [4,4,4],
        [10,10],
        [10]
    ]
    model = FeatureShuffleNet(bottleneck, channels = 64, numofblocks = numofblocks, groups = 32)
    print("FSNeXt_M parameter size: ", count_parameters(model))
    if pretrained:
        # load_path = "./pretrained/triHRnet_Synth_weight.pth"
        # cpt = torch.load(load_path)
        # model.load_state_dict(cpt, strict=True)
        # print("load pretrain weight from {}. ".format(load_path))
        print("FSNeXt_M does not have pretrained weight yet. ")
    return model

def FSNet_S(pretrained = True):
    numofblocks = [
        [4,1,1,1],
        [4,2,2],
        [8,8],
        [4]
    ]
    model = FeatureShuffleNet(basisblock, channels = 64, numofblocks = numofblocks)
    print("FSNet_S parameter size: ", count_parameters(model))
    if pretrained:
        # load_path = "./pretrained/triHRnet_Synth_weight.pth"
        # cpt = torch.load(load_path)
        # model.load_state_dict(cpt, strict=True)
        # print("load pretrain weight from {}. ".format(load_path))
        print("FSNet_S does not have pretrained weight yet. ")
    return model

def FSNeXt_S(pretrained = True):
    numofblocks = [
        [4,1,1,1],
        [4,2,2],
        [8,8],
        [4]
    ]
    model = FeatureShuffleNet(bottleneck, channels = 128, numofblocks = numofblocks, groups = 32)
    print("FSNeXt_S parameter size: ", count_parameters(model))
    if pretrained:
        # load_path = "./pretrained/triHRnet_Synth_weight.pth"
        # cpt = torch.load(load_path)
        # model.load_state_dict(cpt, strict=True)
        # print("load pretrain weight from {}. ".format(load_path))
        print("FSNeXt_S does not have pretrained weight yet. ")
    return model

def FSNet_T(pretrained = True):
    numofblocks = [
        [1,1,1,1],
        [2,1,1],
        [3,3],
        [3]
    ]
    model = FeatureShuffleNet(basisblock, channels = 64, numofblocks = numofblocks)
    print("FSNet_T parameter size: ", count_parameters(model))
    if pretrained:
        # load_path = "./pretrained/triHRnet_Synth_weight.pth"
        # cpt = torch.load(load_path)
        # model.load_state_dict(cpt, strict=True)
        # print("load pretrain weight from {}. ".format(load_path))
        print("FSNet_T does not have pretrained weight yet. ")
    return model