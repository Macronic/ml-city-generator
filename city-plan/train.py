import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from generator import Generator
from discriminator import  Discriminator

from fid_metric import calculate_fid
from inception_metric import inception_score
from kid_metric import calculate_kid
from DISTS_pytorch import DISTS


class SketchCityTrainer():
    def __init__(
        self, 
        channels: int,
        img_size: int,
        num_classes: int,
        seed: int,
        num_workers: int = 2,
        save_root_dir: str = './checkpoints',
        data_dir: str = './generate/',
        log_dir: str = './logs',
        load_checkpoint: str = None,
        batch_kernel: int = 50,
        lambda_gp: int = 10,
        decay: float = 0.005,
        n_critic: int = 5,
        latent_dim: int = 100,
        num_epochs = 600,
        lr: float = 0.0002,
        batch_size: int = 8,
        sample_interval: int = 400,
        *args, 
        **kwargs
    ):

        self.SEED = seed
        self.img_shape = (channels, img_size, img_size)
        self.save_root_dir = save_root_dir
        self.batch_size = batch_size
        self.lambda_gp = lambda_gp
        self.decay = decay
        self.n_critic = n_critic
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.sample_interval = sample_interval
        self.lr = lr
        self.load = False

        if self.save_root_dir is not None and not os.path.exists(self.save_root_dir):
            os.makedirs(self.save_root_dir)
        
        #set seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'CUDA use? {self.device}')
        
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        #dataset
        self.create_dataset(batch_size, img_size, data_dir, num_workers)

        # networks
        gen_modules = [
            (self.latent_dim, 4, 1, 0),
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
            batch_size=batch_size,
            batch_kernel=batch_kernel
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


        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0)
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        self.logger = SummaryWriter(log_dir=log_dir)
        example_inputG_array = torch.zeros(2, self.latent_dim, 1, 1)
        example_inputD_array = torch.zeros(2, *self.img_shape)

        self.logger.add_graph(self.discriminator, example_inputD_array)
        self.logger.add_graph(self.generator, example_inputG_array)

        if torch.cuda.device_count() > 1:
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)
        
        try:
            self.dists = DISTS()
        except Exception as e:
            print(f"Cannot import DISTS problem: {e}")
            self.dists = None

    
    def create_dataset(self, batch_size, img_size, data_dir, num_workers):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = dset.ImageFolder(root=data_dir,transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)
    
    def configure_optimizers(self):
        lr = self.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        if self.load:
            opt_d.load_state_dict(self.optimizer_d_state)
            opt_g.load_state_dict(self.optimizer_g_state)

        return opt_g, opt_d 
    
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
    
    def run(self):
        self.discriminator.train().to(self.device)
        self.generator.train().to(self.device)

        optimizer_G, optimizer_D = self.configure_optimizers()
        batches_done = 0
        save_step = -1

        for epoch in range(self.num_epochs):
            with tqdm(self.dataloader, unit="batch") as tepoch:
                for i, (imgs, _) in enumerate(tepoch):
                    imgs = imgs.to(self.device)
                    save_step += 1
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    optimizer_D.zero_grad()

                    # Sample noise as generator input
                    z = torch.empty((imgs.shape[0], self.latent_dim, 1, 1)).normal_(mean=0.0,std=1.0)
                    z = z.type_as(imgs).to(self.device)

                    # Generate a batch of images
                    fake_imgs = self.generator(z)

                    #make noise
                    down = 1-(self.decay*epoch)
                    if  down < 0 :
                        down = 0
                        
                    noise = torch.empty(imgs.shape).normal_(mean=0.0,std=1.0)*down
                    imgs = (imgs+noise.to(self.device)).clamp_(-1,1)

                    # Real images
                    real_validity = self.discriminator(imgs).view(-1,1)
                    # Fake images
                    fake_validity = self.discriminator(fake_imgs).view(-1,1)
                    # Gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)
                    # Adversarial loss
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

                    d_loss.backward()
                    optimizer_D.step()
                    optimizer_G.zero_grad()

                    # Train the generator every n_critic steps
                    if i % self.n_critic == 0:

                        # -----------------
                        #  Train Generator
                        # -----------------

                        # Generate a batch of images
                        fake_imgs = self.generator(z)
                        # Loss measures generator's ability to fool the discriminator
                        # Train on fake images
                        fake_validity = self.discriminator(fake_imgs).view(-1,1)
                        g_loss = -torch.mean(fake_validity)

                        g_loss.backward()
                        optimizer_G.step()

                        # Prepare inception metrics: FID,IS, DIST, KID
                        fid = calculate_fid(fake_imgs, imgs, False, self.batch_size)
                        is_mean, is_std = inception_score(fake_imgs, cuda=True, batch_size=int(self.batch_size/2))
                        kid_mean, kid_std = calculate_kid(fake_imgs, imgs, self.batch_size)
                        if self.dists is not None:
                            with torch.no_grad():
                                dists_value = self.dists(fake_imgs, imgs, require_grad=False, batch_average=True)
                            self.logger.add_scalar('dist', dists_value, save_step)

                        self.logger.add_scalar('IS_mean', is_mean, save_step)
                        self.logger.add_scalar('IS_std', is_std, save_step)
                        self.logger.add_scalar('kid_mean', kid_mean, save_step)
                        self.logger.add_scalar('kid_std', kid_std, save_step)
                        self.logger.add_scalar('fid', fid, save_step)
                        self.logger.add_scalar('Loss/generator', g_loss, save_step)
                        self.logger.add_scalar('Loss/discriminator', d_loss, save_step)


                        # log sampled images
                        if batches_done % self.sample_interval == 0:
                            if fake_imgs.shape[0] >= 20:
                                sample_imgs = fake_imgs[:20]
                            else:
                                sample_imgs = fake_imgs

                            grid = torchvision.utils.make_grid(sample_imgs)
                            self.logger.add_image('generated_images', grid, epoch)

                        #saving
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.generator.state_dict(),
                            'optimizer_state_dict': optimizer_G.state_dict(),
                            'loss': g_loss,
                            'seed': self.SEED
                        }, f'{self.save_root_dir}/generator-{epoch}')

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.discriminator.state_dict(),
                            'optimizer_state_dict': optimizer_D.state_dict(),
                            'loss': d_loss,
                            'seed': self.SEED
                        }, f'{self.save_root_dir}/discriminator-{epoch}')

                        tepoch.set_postfix_str(
                            f"[Epoch {epoch}] [D loss: {d_loss.item():.5f}] [G loss: {g_loss.item():.5f}]"
                        )

                        batches_done += self.n_critic
        
        self.logger.close()



def main(hparams):
    SEED = 1998
    
    trainer = SketchCityTrainer(
        3,224,1,SEED
    )
    
    trainer.run()

if __name__ == '__main__':
    parser = ArgumentParser(description="python3 train.py --log_gpu_memory true --max_epochs=600 --gpus=1")
    parser.add_argument('--save_root_dir', type=str, default='./checkpoints')
    parser.add_argument('--load_checkpoint', type=str)
    parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    
    hparams = parser.parse_args()
    main(hparams)


    