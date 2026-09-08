import torch
import torch.nn as nn
import torch.fft as fft
import h5py
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.ndimage import gaussian_laplace
import lpips, os, json

# ------------------------------------------------
# Generator CNN
# ------------------------------------------------
class GeneratorCNN(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, 1, 3, padding=1)
        )

    def forward(self, x):

        # ensure input is (B,1,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.dim() == 5:
            x = x.squeeze(1)

        return x + self.net(x)


# ------------------------------------------------
# Data Consistency Layer
# ------------------------------------------------
class DataConsistency(nn.Module):

    def forward(self, x, k_under, mask):

        if x.dim() == 3:
            x = x.unsqueeze(1)

        x_k = torch.fft.fft2(x.squeeze(1), norm='ortho')

        x_k = mask * k_under + (1 - mask) * x_k

        x_img = torch.fft.ifft2(x_k, norm='ortho')

        return x_img.real.unsqueeze(1)


# ------------------------------------------------
# GAN Generator
# ------------------------------------------------
class GANGenerator(nn.Module):

    def __init__(self, iterations=5):

        super().__init__()

        self.iterations = iterations

        self.cnn = GeneratorCNN()

        self.dc = DataConsistency()

    def forward(self, x, k_under, mask):

        if x.dim() == 3:
            x = x.unsqueeze(1)

        for _ in range(self.iterations):

            x = self.cnn(x)

            x = self.dc(x, k_under, mask)

        return x


# ------------------------------------------------
# Discriminator
# ------------------------------------------------
class Discriminator(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channels, channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channels*2, channels*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channels*4, 1, 4, padding=1)
        )

    def forward(self, x):

        # ensure input is (B,1,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.dim() == 5:
            x = x.squeeze(1)

        return self.net(x)
    
def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = GANGenerator().to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr)

    l1_loss = nn.L1Loss()
    adv_loss = nn.BCEWithLogitsLoss()

    lambda_l1 = 100
    lambda_adv = 1

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/gan_recon", exist_ok=True)

    history = {
        "d_loss": [],
        "g_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    best_psnr = 0

    for epoch in range(epochs):

        G.train()
        D.train()

        epoch_d_loss = 0
        epoch_g_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            inp = batch['img_zf'].to(device, non_blocking=True).squeeze(1)
            tgt = batch['target'].to(device, non_blocking=True).squeeze(1)
            k_under = batch['kspace_under'].to(device, non_blocking=True).squeeze(1)
            mask = batch['mask'].to(device, non_blocking=True).squeeze(1)

            # ---------------------------------
            # Train Discriminator
            # ---------------------------------
            opt_D.zero_grad()

            fake = G(inp, k_under, mask).detach()

            real_pred = D(tgt)
            fake_pred = D(fake)

            real_loss = adv_loss(real_pred,
                                 torch.ones_like(real_pred))
            fake_loss = adv_loss(fake_pred,
                                 torch.zeros_like(fake_pred))

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            opt_D.step()

            # ---------------------------------
            # Train Generator
            # ---------------------------------
            opt_G.zero_grad()

            fake = G(inp, k_under, mask)

            fake_pred = D(fake)

            g_adv = adv_loss(fake_pred,
                             torch.ones_like(fake_pred))

            g_l1 = l1_loss(fake, tgt)

            g_loss = lambda_l1 * g_l1 + lambda_adv * g_adv

            g_loss.backward()
            opt_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)

        history["d_loss"].append(epoch_d_loss)
        history["g_loss"].append(epoch_g_loss)

        # ---------------------------------
        # VALIDATION
        # ---------------------------------
        G.eval()

        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Validation"):

                inp = batch['img_zf'].to(device).squeeze(1)
                tgt = batch['target'].to(device).squeeze(1)
                k_under = batch['kspace_under'].to(device).squeeze(1)
                mask = batch['mask'].to(device).squeeze(1)

                out = G(inp, k_under, mask)

                out_np = out.cpu().numpy()
                tgt_np = tgt.cpu().numpy()

                for i in range(out_np.shape[0]):

                    pred = out_np[i].squeeze()
                    gt = tgt_np[i].squeeze()

                    psnr_scores.append(
                        compare_psnr(gt, pred,
                                     data_range=gt.max()-gt.min()+1e-8)
                    )

                    ssim_scores.append(
                        ssim(gt, pred, data_range=1)
                    )

        val_psnr = np.mean(psnr_scores)
        val_ssim = np.mean(ssim_scores)

        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # ---------------------------------
        # Save Best Model
        # ---------------------------------
        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                G.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/gan_recon/best_generator.pt"
            )

        # Save checkpoint each epoch
        if((epoch+1)%10==0):
            torch.save(
                G.state_dict(),
                f"/datasets/tir-datasets-mri-recon/results/checkpoints/gan_recon/epoch_{epoch+1}.pt"
            )


        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"D Loss: {epoch_d_loss:.4f} | "
            f"G Loss: {epoch_g_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # ---------------------------------
    # Save History
    # ---------------------------------
    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/gan_recon/history.json", "w") as f:
        json.dump(history, f)

    return G

def evaluate_model(model, loader):

    device = next(model.parameters()).device
    lpips_model = lpips.LPIPS(net='alex').to(device)

    results = {
        "NMSE":[], "PSNR":[], "SSIM":[],
        "HFEN":[], "VIF":[], "LPIPS":[], "TIME":[]
    }

    model.eval()

    with torch.no_grad():
        for batch in loader:

            inp = batch['img_zf'].to(device).squeeze(1)
            tgt = batch['target'].to(device).squeeze(1)
            k_under = batch['kspace_under'].to(device).squeeze(1)
            mask = batch['mask'].to(device).squeeze(1)

            start = time.time()
            out = model(inp, k_under, mask)
            recon_time = time.time() - start

            for i in range(out.shape[0]):

                pred = out[i].squeeze().cpu().numpy()
                gt   = tgt[i].squeeze().cpu().numpy()

                pred = pred / (pred.max() + 1e-8)
                gt   = gt   / (gt.max() + 1e-8)

                nmse_val = np.linalg.norm(pred-gt)**2 / (np.linalg.norm(gt)**2 + 1e-8)

                results["NMSE"].append(nmse_val)
                results["PSNR"].append(compare_psnr(gt, pred, data_range=1))
                results["SSIM"].append(ssim(gt, pred, data_range=1, win_size=7))

                results["HFEN"].append(
                    np.linalg.norm(
                        gaussian_laplace(pred,1.5) -
                        gaussian_laplace(gt,1.5)
                    ) / (np.linalg.norm(gaussian_laplace(gt,1.5)) + 1e-8)
                )

                results["VIF"].append(
                    np.var(pred)/(np.var(gt)+1e-8)
                )

                pred_lp = torch.tensor(pred[None,None,:,:]).repeat(1,3,1,1).float().to(device)
                gt_lp   = torch.tensor(gt[None,None,:,:]).repeat(1,3,1,1).float().to(device)

                pred_lp = pred_lp * 2 - 1
                gt_lp   = gt_lp * 2 - 1

                lpips_val = lpips_model(pred_lp, gt_lp).mean().item()
                results["LPIPS"].append(lpips_val)

                results["TIME"].append(recon_time)

    return {k: np.mean(v) for k,v in results.items()}