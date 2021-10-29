import torch 
import torch.nn as nn
from torchvision.utils import save_image


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size =4, stride = stride, bias = False, padding_mode = 'reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True)
        )
    def forward(self, x):
        return self.conv(x) 

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.LeakyReLU(0.2, inplace = True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
               CNNBlock(in_channels, feature, stride = 1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect'
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim = 1)
        x = self.initial(x)
        return self.model(x)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

def test():
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3,256, 256)
    model = Discriminator()
    model.apply(init_weights)
    preds = model(x,y)
    print(preds.shape)
    save_image(preds, "d.png")

if __name__ == '__main__':
    test()
