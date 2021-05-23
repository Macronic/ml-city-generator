import torch
import torch.utils.data

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image
import os


class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cfg: str):
        self.cfg = cfg
        self.image_paths = []
        self.target_paths = []

        self.map = []
        for folder in os.listdir(cfg.dataset):
            if 'map' in folder:
                self.map.append(os.path.join(cfg.dataset, folder))
        
        for mape in self.map:
            for maper in os.listdir(mape):
                if 'png' in maper and 'georef' not in maper:
                    satelite = mape.replace('map', 'satelitte')
                    sateliter = maper.replace('png', 'jpg')
                    condtion = os.path.join(satelite, sateliter)
                    if os.exists(condtion):
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
        if self.cfg.channels = 3:
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask = TF.normalize(mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.cfg.channels = 1:
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask = TF.normalize(mask, mean=[0.5], std=[0.5])

        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)
    

