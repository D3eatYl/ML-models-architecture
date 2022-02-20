# resnet implementation
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        # shortcut
        # the shortcut output dimension is not the same with the residual function
        # use 1*1 convolution to match the dimension
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=BasicBlock.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
            )
    def froward(self, x):
        identity = self.shortcut(x)
        residual = self.residual_function(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = residual+identity
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    # Residual block for over 50 layers
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        # shortcut
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=out_channels*BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.residual_function(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = residual+identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # there is different from the original paper
        # original paper: conv2 = self._make_layer(block, 64, num_block[0], stride=2)
        # original paper without conv5
        self.conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.conv4 = self._make_layer(block, 256, num_block[2], 2)
        self.conv5 = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride]+[1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet10():
    return ResNet(BasicBlock, [1, 1, 1, 1])

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])
