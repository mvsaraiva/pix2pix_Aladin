import torch 
import torch.nn as nn
from torchvision.utils import save_image
import torch.optim as optim
from PIL import Image
import numpy as np
from utils import load_checkpoint
import config
from discriminator_model import Discriminator

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

def test():
    
    x = torch.Tensor(np.array(Image.open('y.png'))[0:256,0:256,:]).unsqueeze(0).permute(0,3,1,2)
    print(x.shape)
    y = torch.Tensor(np.array(Image.open('y.png'))[0:256,0:256,:]).unsqueeze(0).permute(0,3,1,2)
    model = Discriminator()
    opt_disc = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    load_checkpoint(
            config.CHECKPOINT_DISC, model, opt_disc, config.LEARNING_RATE,
        )
    model.apply(init_weights)
    preds = model(x,y)
    print(preds.shape)
    save_image(preds, "d.png")

if __name__ == '__main__':
    test()
