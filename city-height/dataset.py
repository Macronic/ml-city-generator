import torch
import torch.utils.data

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt

import os
import random
import yaml

from easydict import EasyDict as edict


class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cfg: str, seed: int):
        self.cfg = edict(cfg)
        random.seed(seed)
        self.image_paths = []
        self.target_paths = []

        self.map = []
        for folder in os.listdir(self.cfg.dataset):
            if 'satelitte-hm' in folder:
                self.map.append(os.path.join(self.cfg.dataset, folder))
        
        for mape in self.map:
            for maper in os.listdir(mape):
                if 'jpg' in maper and 'georef' not in maper:
                    satelite = mape.replace('satelitte', 'heightmap')
                    sateliter = maper.replace('jpg', 'png')
                    condtion = os.path.join(satelite, sateliter)
                    if os.path.exists(condtion):
                        self.image_paths.append(os.path.join(mape, maper))
                        self.target_paths.append(condtion)
        
    def transform(self, image, mask):
        # Resize
        if self.cfg.develop:
            size = 128
        else:
            size = 256

        resize = transforms.Resize(size=(size, size))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(size, size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        #normalize
        if self.cfg.channels == 3:
            image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            mask = TF.normalize(mask, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.cfg.channels == 1:
            image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            mask = TF.normalize(mask, mean=[0.5], std=[0.5])

        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.target_paths[index]).convert("L")
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)
    

if __name__=="__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset = DatasetTrain(cfg['data'], cfg['seed'])
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        condition, real = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title("REAL")
        plt.axis("off")
        plt.imshow(real.squeeze().mul(0.5).add(0.5).cpu().permute(1,2,0).numpy())
    
    #plt.savefig("sample_loader.png")
    plt.show()

