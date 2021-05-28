#!/bin/python
'''
Insparied https://librecv.github.io/blog/gans/pytorch/2021/02/13/Pix2Pix-explained-with-code.html
'''
import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

import os
import yaml

from datetime import datetime 

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
from torch import nn
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import DatasetTrain
from generator import Generator
from discriminator import Discriminator

from fid_metric import calculate_fid
from inception_metric import inception_score
from kid_metric import calculate_kid
from DISTS_pytorch import DISTS


class DrawCityGAN(LightningModule):
    def __init__(self, config_path: str, **kwargs):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        super().__init__()
        self.save_hyperparameters(cfg)
        pl.seed_everything(self.hparams.seed)
        if self.hparams.develop:
            self.tags = ['CityGeneration', "DrawGeneration", "develop"]
        else:
            self.tags = ['CityGeneration', "DrawGeneration", "production"]

        # networks
        self.generator = Generator(config=self.hparams.generator)
        self.discriminator = Discriminator(config=self.hparams.discriminator)
        
        # loss
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

        if self.hparams.develop:
            self.img_size = 256
        else:
            self.img_size = 512

        if self.hparams.noise_decrease != 0:
            self.noise_img = 1
        else:
            self.noise_img = 0

        self.temp_logger = SummaryWriter(log_dir=self.hparams.temp_log_dir, comment=datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))

        try:
            self.dists = DISTS()
        except Exception as e:
            print(f"Cannot import DISTS problem: {e}")
            self.dists = None
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        self.fake_images = self.generator(conditioned_images)
        disc_logits = self.discriminator(self.fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(self.fake_images, real_images)
        #lambda_recon = self.hparams.lambda_recon
        
        del disc_logits
        #return adversarial_loss + lambda_recon * recon_loss
        return adversarial_loss, recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.generator(conditioned_images).detach()
        fake_logits = self.discriminator(fake_images, conditioned_images)

        real_logits = self.discriminator(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        del fake_images, fake_logits, real_logits

        #return (real_loss + fake_loss) / 2
        return real_loss, fake_loss
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lrg, betas=(self.hparams.b1g, self.hparams.b2g))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lrd, betas=(self.hparams.b1d, self.hparams.b2d))
        return opt_d, opt_g

    def train_dataloader(self):
        dataset = DatasetTrain(self.hparams.data, self.hparams.seed)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.workers, pin_memory=True)
    
    def training_step(self, batch, batch_idx):
        condition, real = batch

        # add noise to images
        noise = torch.empty(real.shape, device=self.device).normal_(mean=0.0,std=1.0)*self.noise_img
        real = (real+noise).clamp_(-1,1)
        condition = (condition+noise).clamp_(-1,1)
        
        d_opt, g_opt = self.optimizers()
        
        # ----------------
        # Train generator
        # ---------------
        adv, recon = self._gen_step(real, condition)
        loss = adv + self.hparams.lambda_recon * recon
        
        g_opt.zero_grad()
        self.manual_backward(loss)
        g_opt.step()

        self.logger.experiment.log_metric('Generator/adverserial_loss', adv, self.global_step, self.current_epoch)
        self.logger.experiment.log_metric('Generator/reconstrution_loss', recon, self.global_step, self.current_epoch)
        self.logger.experiment.log_metric('Generator/loss', loss, self.global_step, self.current_epoch)
        
        # -------------------
        # Train discriminator
        # -------------------
        real_loss, fake_loss = self._disc_step(real, condition)
        loss = (real_loss+fake_loss)/2

        d_opt.zero_grad()
        self.manual_backward(loss)
        d_opt.step()
        
        self.logger.experiment.log_metric('Discriminator/real_loss', real_loss, self.global_step, self.current_epoch)
        self.logger.experiment.log_metric('Discriminator/fake_loss', fake_loss, self.global_step, self.current_epoch)
        self.logger.experiment.log_metric('Discriminator/loss', loss, self.global_step, self.current_epoch)
    

        self.logger.experiment.log_metric('Noise', self.noise_img, self.global_step, self.current_epoch)
        try:
            self.logger.experiment.log_metric('FID', calculate_fid(self.fake_images, real.detach(), False, 1), self.global_step, self.current_epoch)
        except Exception as e:
            print(f"Exception in fid metric {e}")
        try:
            self.logger.experiment.log_metric('IS/mean', inception_score(self.fake_images, cuda=False, batch_size=1)[0], self.global_step, self.current_epoch)
        except Exception as e:
            print(f"Exception in IS metric {e}")
        try:
            self.logger.experiment.log_metric('KID/mean', calculate_kid(self.fake_images, real.detach(), 1)[0], self.global_step, self.current_epoch)
        except Exception as e:
            print(f"Exception in kid metric {e}")
        if self.dists is not None:
            try:
                with torch.no_grad():
                    self.logger.experiment.log_metric('DISTS', self.dists(self.fake_images, real.detach(), require_grad=False, batch_average=True), self.global_step, self.current_epoch)
            except Exception as e:
                print(f"Exception in dists metric {e}")

        if self.global_step%self.hparams.display_step==0 and self.hparams.display_show:
            grid = torchvision.utils.make_grid(self.fake_images[:4])
            self.temp_logger.add_image('generated_images', grid, self.global_step)

            grid = torchvision.utils.make_grid(condition[:4].detach())
            self.temp_logger.add_image('condition_images', grid, self.global_step)

            grid = torchvision.utils.make_grid(real[:4].detach())
            self.temp_logger.add_image('real_images', grid, self.global_step)
        
        return loss

    def on_epoch_end(self):
        self.noise_img -= self.hparams.noise_decrease
        if self.noise_img < 0:
            self.noise_img = 0

        if self.tags is not None:
            self.logger.experiment.add_tags(self.tags)
            self.tags = None

        if self.current_epoch % self.hparams.interval_save == 0 and self.hparams.interval_show:
            grid = torchvision.utils.make_grid(self.fake_images[:4]).mul(0.5).add(0.5).cpu().permute(1,2,0).numpy()
            try:
                self.logger.experiment.log_image(grid, name='generated_images', step=self.current_epoch)
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
    model = DrawCityGAN(config_path=args.config_path)

    # ------------------------
    # 2 INIT LOGGER 
    # ------------------------
    comet_logger = CometLogger(
        api_key="***REMOVED***",
        project_name="citygeneration",
        workspace="wolodja"
    )

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(logger=comet_logger, gpus=args.gpus, max_epochs=args.epochs)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="config path")

    hparams = parser.parse_args()

    main(hparams)
