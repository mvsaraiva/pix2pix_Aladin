import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act = 'relu', use_dropout = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False, padding_mode = 'reflect')
            if down 
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2, inplace = True)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels, features, kernel_size = 4, act = 'relu', use_dropout = False):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.LeakyReLU(0.2, inplace = True),
        )

        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout = False)
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout = False)
        self.down3 = Block(features*4, features*8, down=True, act="leaky", use_dropout = False)
        self.down4 = Block(features*8, features*8, down=True, act="leaky", use_dropout = False)
        self.down5 = Block(features*8, features*8, down=True, act="leaky", use_dropout = False)
        self.down6 = Block(features*8, features*8, down=True, act="leaky", use_dropout = False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.ReLU(),
        )
        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout = True)
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout = True)
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout = True)
        self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout = False)
        self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout = False)
        self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout = False)
        self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout = False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features *2, in_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bn = self.bottleneck(d7)
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d7], dim = 1))
        u3 = self.up3(torch.cat([u2, d6], dim = 1))
        u4 = self.up4(torch.cat([u3, d5], dim = 1))
        u5 = self.up5(torch.cat([u4, d4], dim = 1))
        u6 = self.up6(torch.cat([u5, d3], dim = 1))
        u7 = self.up7(torch.cat([u6, d2], dim = 1))
        return self.final_up(torch.cat([u7, d1], dim = 1))

def test():
    x = torch.randn(1, 3, 256, 256)
    x_d6 =torch.randn(1, 64*8,2,2)

    model = Generator(in_channels=3, features=64)
    preds = model(x)
    bottle_neck = model.bottleneck(x_d6)
    print(preds.shape)
    print(bottle_neck.shape)

if __name__ == '__main__':
    test()

