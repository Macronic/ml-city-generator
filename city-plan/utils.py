
import math
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models

from numpy.linalg import norm
from scipy import linalg
from scipy.stats import entropy


def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_real = s.tp / (s.tp + s.fn)
    s.acc_fake = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd

def compute_score(real, fake, k=1, sigma=1, sqrt=True):

    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    emd = wasserstein(Mxy, sqrt)
    mmd = mmd(Mxx, Mxy, Myy, sigma)
    knn = knn(Mxx, Mxy, Myy, k, sqrt)

    return emd, mmd, knn.precision, knn.recall

class FID():
    '''
    Code for FID Calculation taken from TA's piazza post
    '''
    def __init__(self, cache_dir='./Cache', device='cpu', transform_input=True):
        os.environ["TORCH_HOME"] = "./Cache"
        self.device=device
        self.transform_input = transform_input
        self.InceptionV3 = models.inception_v3(pretrained=True, transform_input=False, aux_logits=False).to(device=self.device)
        self.InceptionV3.eval()
    
    def build_maps(self, x):
        # Resize to Fit InceptionV3
        if list(x.shape[-2:]) != [299,299]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = F.interpolate(x, size=[299,299], mode='bilinear')
        # Transform Input to InceptionV3 Standards
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # Run Through Partial InceptionV3 Model
        with torch.no_grad():
            # N x 3 x 299 x 299
            x = self.InceptionV3.Conv2d_1a_3x3(x)
            # N x 32 x 149 x 149
            x = self.InceptionV3.Conv2d_2a_3x3(x)
            # N x 32 x 147 x 147
            x = self.InceptionV3.Conv2d_2b_3x3(x)
            # N x 64 x 147 x 147
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # N x 64 x 73 x 73
            x = self.InceptionV3.Conv2d_3b_1x1(x)
            # N x 80 x 73 x 73
            x = self.InceptionV3.Conv2d_4a_3x3(x)
            # N x 192 x 71 x 71
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # N x 192 x 35 x 35
            x = self.InceptionV3.Mixed_5b(x)
            # N x 256 x 35 x 35
            x = self.InceptionV3.Mixed_5c(x)
            # N x 288 x 35 x 35
            x = self.InceptionV3.Mixed_5d(x)
            # N x 288 x 35 x 35
            x = self.InceptionV3.Mixed_6a(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6b(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6c(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6d(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6e(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_7a(x)
            # N x 1280 x 8 x 8
            x = self.InceptionV3.Mixed_7b(x)
            # N x 2048 x 8 x 8
            x = self.InceptionV3.Mixed_7c(x)
            # N x 2048 x 8 x 8
            # Adaptive average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))
            # N x 2048 x 1 x 1
        
        return x
    
    def forward_model(self, x):
        if list(x.shape[-2:]) != [299,299]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = F.interpolate(x, size=[299,299], mode='bilinear')
        # Transform Input to InceptionV3 Standards
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # Run Through Partial InceptionV3 Model
        with torch.no_grad():
            x = self.InceptionV3(x)
        
        return F.softmax(x, dim=1).data.cpu().numpy()

    def compute_fid(self, real_images, generated_images, batch_size=64):
        # Ensure Set Sizes are the Same
        assert(real_images.shape[0] == generated_images.shape[0])
        # Build Random Sampling Orders
        real_images = real_images[np.random.permutation(real_images.shape[0])]
        generated_images = generated_images[np.random.permutation(generated_images.shape[0])]
        # Lists of Maps per Batch
        real_maps = []
        generated_maps = []
        # Build Maps
#        for s in tqdm(range(int(math.ceil(real_images.shape[0]/batch_size))), desc='Evaluation', leave=False):
        for s in range(int(math.ceil(real_images.shape[0]/batch_size))):
            sidx = np.arange(batch_size*s, min(batch_size*(s+1), real_images.shape[0]))
            real_maps.append(self.build_maps(real_images[sidx].to(device=self.device)).detach().to(device='cpu'))
#            real_maps.append(self.build_maps(real_images[sidx]).detach())
            generated_maps.append(self.forward_model(generated_images[sidx].to(device=self.device)).detach().to(device='cpu'))
#            generated_maps.append(self.build_maps(generated_images[sidx]).detach())
        # Concatenate Maps
        real_maps = np.squeeze(torch.cat(real_maps).numpy())
        generated_maps = np.squeeze(torch.cat(generated_maps).numpy())
        # Calculate IS
        # Activation Statistics
        mu_g = np.mean(generated_maps, axis=0)
        mu_x = np.mean(real_maps, axis=0)
        sigma_g = np.cov(generated_maps, rowvar=False)
        sigma_x = np.cov(real_maps, rowvar=False)
        # Sum of Squared Differences
        ssd = np.sum((mu_g - mu_x)**2)
        # Square Root of Product of Covariances
        covmean = linalg.sqrtm(sigma_g.dot(sigma_x), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Final FID Computation
        return ssd + np.trace(sigma_g + sigma_x - 2*covmean)
    
    def compute_IS(self, generated_images, batch_size=64, splits=1):
        N = generated_images.shape[0]
        generated_images = generated_images[np.random.permutation(generated_images.shape[0])]
        # Lists of Maps per Batch
        preds = np.array([])
        # Build Maps
#        for s in tqdm(range(int(math.ceil(real_images.shape[0]/batch_size))), desc='Evaluation', leave=False):
        for s in range(int(math.ceil(real_images.shape[0]/batch_size))):
            sidx = np.arange(batch_size*s, min(batch_size*(s+1), generated_images.shape[0]))
            preds[sidx] = self.build_maps(generated_images[sidx].to(device=self.device))
#            generated_maps.append(self.build_maps(generated_images[sidx]).detach())
        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

        # Calculate FID
        # Activation Statistics
        mu_g = np.mean(generated_maps, axis=0)
        mu_x = np.mean(real_maps, axis=0)
        sigma_g = np.cov(generated_maps, rowvar=False)
        sigma_x = np.cov(real_maps, rowvar=False)
        # Sum of Squared Differences
        ssd = np.sum((mu_g - mu_x)**2)
        # Square Root of Product of Covariances
        covmean = linalg.sqrtm(sigma_g.dot(sigma_x), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Final FID Computation
        return ssd + np.trace(sigma_g + sigma_x - 2*covmean)


#https://github.com/vict0rsch/pytorch-fid-wrapper
#https://github.com/KarthiVi95/DCGAN_Pytorch/blob/master/DCGAN.ipynb
#https://github.com/enijkamp/metrics_generative/blob/master/inception_score_v3_torch.py