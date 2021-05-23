#!/bin/python
'''
Insparied https://librecv.github.io/blog/gans/pytorch/2021/02/13/Pix2Pix-explained-with-code.html
'''
import os
import yaml

from datetime import datetime 

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import torchvision

from torch.utils.data import DataLoader

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from dataset import DatasetTrain

class DrawCityGAN(LightningModule):
    def __init__(self, config_path: str, **kwargs):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        super().__init__()
        self.save_hyperparameters(cfg)
        if self.hparams.develop:
            self.logger.experiment.add_tags(['CityGeneration', "DrawGeneration", "develop"])
        else:
            self.logger.experiment.add_tags(['CityGeneration', "DrawGeneration", "production"])

        # networks
        self.generator = Generator(cfg=self.hparams.generator)
        self.discriminator = Discriminator(cfg=self.hparams.discriminator)
        
        # loss
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

        if self.hparams.develop:
            self.img_size = 256
        else:
            self.img_size = 512

        self.hparams.noise_decrease != 0:
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
        fake_images = self.generator(conditioned_images)
        disc_logits = self.discriminator(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.generator(conditioned_images).detach()
        fake_logits = self.discriminator(fake_images, conditioned_images)

        real_logits = self.discriminator(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lrg, betas=(self.hparams.b1g, self.hparams.b2g))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lrd, betas=(self.hparams.b1d, self.hparams.b2d))
        return opt_d, opt_g

    def train_dataloader(self):
        dataset = DatasetTrain(self.hparams.data)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, condition = batch

        # add noise to images
        noise = torch.empty(real.shape).normal_(mean=0.0,std=1.0)*self.noise_img
        real = (real+noise.to(self.device)).clamp_(-1,1)
        condition = (condition+noise.to(self.device)).clamp_(-1,1)

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
            log_dict = {'Loss/PatchGAN-Discriminator': loss}

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
            
            self.log_dict(log_dict, prog_bar=True)

        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
            self.log('Loss/Generator', loss)
        
        if self.current_epoch%self.hparams.display_step==0 and batch_idx==0 and optimizer_idx==1:
            fake = self.gen(condition).detach()
            grid = torchvision.utils.make_grid(fake)
            self.temp_logger.add_image('generated_images', grid, 0)

            grid = torchvision.utils.make_grid(condition)
            self.temp_logger.add_image('condition_images', grid, 0)

            grid = torchvision.utils.make_grid(real)
            self.temp_logger.add_image('real_images', grid, 0)
        
        self.condition = condition
        return loss

    def on_epoch_end(self):
        self.noise_img -= self.hparams.noise_decrease
        if self.current_epoch % self.hparams.interval_save == 0:
            fake = self.gen(self.condition).detach()
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