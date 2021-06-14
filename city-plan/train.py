#!/bin/python
import os
import yaml

from datetime import datetime 

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

from torch.utils.data import DataLoader

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

class SketchCityGAN(LightningModule):
    def __init__(self, config_path: str, **kwargs):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        super().__init__()
        self.save_hyperparameters(cfg)

        # networks
        self.generator = Generator(cfg=self.hparams.generator)
        self.discriminator = Discriminator(cfg=self.hparams.discriminator)

        self.validation_z = torch.empty((8, self.hparams.generator.latent_dim, 1, 1)).normal_(mean=0.0,std=1.0)
        self.example_input_array = torch.empty((2, self.hparams.generator.latent_dim, 1, 1)).normal_(mean=0.0,std=1.0)

        if self.hparams.develop:
            self.img_size = 256
        else:
            self.img_size = 512

        self.noise_img = 0
        self.temp_logger = SummaryWriter(log_dir=self.hparams.temp_log_dir, comment=datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))

        try:
            self.dists = DISTS()
        except Exception as e:
            print(f"Cannot import DISTS problem: {e}")
            self.dists = None
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        # add noise to images
        noise = torch.empty(imgs.shape).normal_(mean=0.0,std=1.0)*self.noise_img
        imgs = (imgs+noise.to(self.device)).clamp_(-1,1)

        # sample noise
        z = torch.empty((imgs.shape[0], self.hparams.generator.latent_dim, 1, 1)).normal_(mean=0.0,std=1.0)
        z = z.type_as(imgs)

        g_opt, d_opt = self.optimizers()
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in self.discriminator.parameters():
            p.requires_grad = True

        self.discriminator.zero_grad()

        # Generate a batch of images
        fake_imgs = self(z)

        # Real images
        real_validity = self.discriminator(imgs)
        self.manual_backward(real_validity, gradient=mone)
        # Fake images
        fake_validity = self.discriminator(fake_imgs)
        self.manual_backward(fake_validity, gradient=one)
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)
        self.manual_backward(gradient_penalty)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.hparams.lambda_gp * gradient_penalty
        d_opt.step()

        log_dict = {
            'Loss/discriminator': d_loss,
            'Loss/discriminator/real': real_validity.data,
            'Loss/discriminator/fake': fake_validity.data,
            'Wasserstain-distance': real_validity.data-fake_validity.data
        }

        try:
            log_dict['FID'] = calculate_fid(fake_imgs.cpu().detach(), imgs.cpu().detach(), False, 2)
        except:
            pass

        try:
            log_dict['IS/mean'], _ = inception_score(fake_imgs.cpu().detach(), cuda=False, batch_size=2)
        except:
            pass

        try:
            log_dict['KID/mean'], _ = calculate_kid(fake_imgs, imgs, 2)
        except:
            pass

        if self.dists is not None:
            try:
                with torch.no_grad():
                    log_dict['DISTS'] = self.dists(fake_imgs, imgs, require_grad=False, batch_average=True)
            except:
                pass

        # -----------------
        #  Train Generator
        # -----------------
        if batch_idx % self.hparams.n_critic == 0:
            for p in self.discriminator.parameters():
                p.requires_grad = False
                                
            self.generator.zero_grad()
            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.temp_logger.add_image('generated_images', grid, 0)

            sample_imgs = imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.temp_logger.add_image('real_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = -torch.mean(self.discriminator(self(z)))
            self.manual_backward(g_loss, gradient=mone)
            g_opt.step()

            log_dict['Loss/generator']: g_loss

        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lrg, betas=(self.hparams.b1g, self.hparams.b2g))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lrd, betas=(self.hparams.b1d, self.hparams.b2d))
        return opt_g, opt_d

    def train_dataloader(self):
        if self.hparams.channels == 3:
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif self.hparams.channels == 1:
            transform = transforms.Compose([
                transforms.CenterCrop(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
        dataset = dset.ImageFolder(root=self.hparams.data_dir,transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def on_epoch_end(self):
        if self.current_epoch % self.hparams.interval_save == 0:
            z = self.validation_z.to(self.device)

            # log sampled images
            sample_imgs = self(z)
            grid = torchvision.utils.make_grid(sample_imgs).mul(0.5).add(0.5).cpu().permute(1,2,0).detach().numpy()
            try:
                self.logger.experiment.log_image('generated_images', grid, self.current_epoch)
            except Exception as e:
                print(e)
            
            # log weights
            weights = []
            for name in self.generator.named_parameters():
                if 'weight' in name[0]:
                    weights.extend(name[1].cpu().detach().numpy().tolist())
            
            try:
                self.logger.experiment.log_histogram_3d(weights, step=self.current_epoch)
            except Exception as e:
                print(e)


def main(args: Namespace) -> None:
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = WGANGP(config_path=args.config_path)

    # ------------------------
    # 2 INIT LOGGER 
    # ------------------------
    comet_logger = CometLogger(
        api_key="",
        project_name="citygeneration",
        workspace="wolodja"
    )

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(logger=CometLogger, gpus=args.gpus)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="config path")

    hparams = parser.parse_args()

    main(hparams)
