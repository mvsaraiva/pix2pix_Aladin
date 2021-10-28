from PIL import Image 
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img_file = self.list_files[idx]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:,:600,:]
        target_image = image[:,600:,:]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

if __name__ == "__main__":
    dataset = MapDataset("/reser/ATP-AB/ALBACORA/GEOFISICA/E_Especiais/pix2pix_Aladin/data/maps/train")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        print(x.min(), x.max())
        print(y.min(), y.max())
        save_image((x+1)/2, "x.png")
        save_image((y+1)/2, "y.png")
        import sys

        sys.exit()


