import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable

from generator import Generator
from discriminator import  Discriminator
#from utils import compute_score_raw

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

class SketchCityDataModule(pl.LightningDataModule):

    def __init__(self, img_size: int, data_dir: str = './generate/', batch_size: int = 64, num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, img_size, img_size)
        self.num_classes = 1

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        dataset = dset.ImageFolder(root=self.data_dir,transform=self.transform)
        self.dataset_train, self.dataset_val = random_split(dataset, 
                            [int(len(dataset)*9/10), 
                            len(dataset)-int(len(dataset)*9/10)]
                        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

class SketchGAN(pl.LightningModule):

    def __init__(
        self,
        channels: int,
        img_size: int,
        num_classes: int,
        seed: int,
        save_root_dir: str = './checkpoints',
        load_checkpoint: str = None,
        batch_kernel: int = 50,
        lambda_gp: int = 10,
        decay: float = 0.005,
        n_critic: int = 5,
        latent_dim: int = 100,
        num_epochs = 600,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 16,
        **kwargs
    ):
        super().__init__()
        self.SEED = seed
        self.save_hyperparameters()
        self.save_root_dir = save_root_dir
        self.load = False

        if self.save_root_dir is not None and not os.path.exists(self.save_root_dir):
            os.makedirs(self.save_root_dir)

        # networks
        gen_modules = [
            (self.hparams.latent_dim, 4, 1, 0),
            (1024, 4, 1, 0),
            (512, 4, 2, 1),
            (256, 4, 2, 1),
            (128, 4, 2, 1),
            (64, 4, 2, 1),
            (32, 4, 2, 1),
            (3, 0, 0, 0)
            #(16, 1, 1, 0),
            #(3, 0, 0, 0)
        ]
        disc_modules = [
            (3, 4, 2, 1),
            (32, 4, 2, 1),
            (64, 4, 2, 1),
            (128, 4, 2, 1),
            (256, 4, 2, 1),
            (512, 4, 2, 1),
            (1024, 3, 1, 0),
            (128, 0, 0, 0)
        ]
        
        self.generator = Generator(module_list=gen_modules)
        self.discriminator = Discriminator(
            classes_num=num_classes,  
            cnn_list=disc_modules, 
            batch_size=self.hparams.batch_size,
            batch_kernel=self.hparams.batch_kernel
        )

        if load_checkpoint is not None:
            checkpoint_dim = torch.load(
                f'{self.save_root_dir}/discriminator-{load_checkpoint}'
            )
            self.discriminator.load_state_dict(checkpoint_dim['model_state_dict'])
            self.optimizer_d_state = checkpoint_dim['optimizer_state_dict']
            epoch_d = checkpoint_dim['epoch']
            loss_d = checkpoint_dim['loss']

            checkpoint_gen = torch.load(
                f'{self.save_root_dir}/generator-{load_checkpoint}'
            )
            self.generator.load_state_dict(checkpoint_gen['model_state_dict'])
            self.optimizer_g_state = checkpoint_gen['optimizer_state_dict']
            epoch_g = checkpoint_gen['epoch']
            loss_g = checkpoint_gen['loss']

            self.load = True
            if loss_g > loss_d:
                current_epoch = loss_g+1
            else:
                current_epoch = loss_d+1
            # TODO set start epoch to current epoch
            print(f"Last generator loss {loss_g}, last discriminator loss {loss_d} on epoch {current_epoch-1}")


        self.validation_z = torch.empty((self.hparams.batch_size, self.hparams.latent_dim, 1, 1)).normal_(mean=0.0,std=1.0)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim, 1, 1)

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0)
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.generator(z)
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates).view(-1,1)
        fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        self.last_imgs = imgs

        # sample noise
        z = torch.empty((imgs.shape[0], self.hparams.latent_dim, 1, 1)).normal_(mean=0.0,std=1.0)
        z = z.type_as(imgs)

        # ---------------------
        #  Train Generator
        # ---------------------
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            if imgs.shape[0] >= 6:
                sample_imgs = self.generated_imgs[:6]
            else:
                sample_imgs = self.generated_imgs
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = -torch.mean(self.discriminator(self(z)).view(-1,1))
            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
            
            opt = (self.optimizers()[0]).optimizer
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': g_loss,
                'seed': self.SEED
            }, f'{self.save_root_dir}/generator-{self.current_epoch}')

        # ---------------------
        #  Train Discriminator
        # ---------------------
        elif optimizer_idx == 1:
            fake_imgs = self(z)

            # Add noise
            down = 1-(self.hparams.decay*self.current_epoch)
            if  down < 0 :
                down = 0
                
            noise = torch.empty(imgs.shape).normal_(mean=0.0,std=1.0)*down
            imgs = (imgs+noise).clamp_(-1,1)
            
            # Real images
            real_validity = self.discriminator(imgs).view(-1,1)
            # Fake images
            fake_validity = self.discriminator(fake_imgs).view(-1,1)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.hparams.lambda_gp * gradient_penalty
            self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
            
            opt = (self.optimizers()[1]).optimizer
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': d_loss,
                'seed': self.SEED
            }, f'{self.save_root_dir}/discriminator-{self.current_epoch}')

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        if self.load:
            opt_d.load_state_dict(self.optimizer_d_state)
            opt_g.load_state_dict(self.optimizer_g_state)

        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': self.hparams.n_critic}
        )

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        ################################################
        #### metric scores computing (key function) ####
        ################################################
        
        #self.logger.experiment.add


def main(hparams):
    SEED = 1998
    pl.seed_everything(SEED)

    dm = SketchCityDataModule(img_size=224)
    model = SketchGAN(**vars(hparams), channels=3, img_size=224, num_classes=1, seed=SEED)
    
    tb_logger = pl_loggers.TensorBoardLogger('./logs/')
    
    trainer = Trainer.from_argparse_args(
        hparams,
        profiler="simple",
        logger=tb_logger
    )
    
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser(description="python3 train.py --log_gpu_memory true --max_epochs=600 --gpus=1")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--save_root_dir', type=str, default='./checkpoints')
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--load_checkpoint', type=str)
    parser = Trainer.add_argparse_args(parser)
    #parser = SketchGAN.add_model_specific_args(parser)
    hparams = parser.parse_args()
    
    main(hparams)


    