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

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DAGANGenerator(nn.Module):

    def __init__(self, base=64):

        super().__init__()

        # encoder
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)

        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = ConvBlock(base*4, base*8)

        # decoder
        self.up3 = UpBlock(base*8, base*4)
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)

        self.final = nn.Conv2d(base, 1, 1)

    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.final(d1)

        return x + out
    
class DAGANDiscriminator(nn.Module):

    def __init__(self, base=64):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(base, base*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(base*2, base*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(base*4, 1, 4, padding=1)
        )

    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)

        return self.net(x)
    
class DataConsistency(nn.Module):

    def forward(self, x, k_under, mask):

        x_k = torch.fft.fft2(x.squeeze(1), norm="ortho")

        x_k = mask * k_under + (1-mask) * x_k

        x_img = torch.fft.ifft2(x_k, norm="ortho")

        return x_img.real.unsqueeze(1)
    
class DAGAN(nn.Module):

    def __init__(self):

        super().__init__()

        self.generator = DAGANGenerator()

        self.dc = DataConsistency()

    def forward(self, x, k_under, mask):

        x = self.generator(x)

        x = self.dc(x, k_under, mask)

        return x
    
def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = DAGAN().to(device)
    D = DAGANDiscriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr)

    l1 = nn.L1Loss()
    adv = nn.BCEWithLogitsLoss()

    lambda_l1 = 100
    lambda_adv = 1

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/dagan", exist_ok=True)

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

            inp = batch["img_zf"].to(device, non_blocking=True).squeeze(1)
            tgt = batch["target"].to(device, non_blocking=True).squeeze(1)
            k = batch["kspace_under"].to(device, non_blocking=True).squeeze(1)
            mask = batch["mask"].to(device, non_blocking=True).squeeze(1)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_D.zero_grad()

            fake = G(inp, k, mask).detach()

            real_pred = D(tgt)
            fake_pred = D(fake)

            real_loss = adv(real_pred, torch.ones_like(real_pred))
            fake_loss = adv(fake_pred, torch.zeros_like(fake_pred))

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            opt_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_G.zero_grad()

            fake = G(inp, k, mask)

            fake_pred = D(fake)

            g_adv = adv(fake_pred, torch.ones_like(fake_pred))
            g_l1 = l1(fake, tgt)

            g_loss = lambda_l1 * g_l1 + lambda_adv * g_adv

            g_loss.backward()
            opt_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)

        history["d_loss"].append(epoch_d_loss)
        history["g_loss"].append(epoch_g_loss)

        # ---------------------
        # Validation
        # ---------------------
        G.eval()

        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Validation"):

                inp = batch["img_zf"].to(device).squeeze(1)
                tgt = batch["target"].to(device).squeeze(1)
                k = batch["kspace_under"].to(device).squeeze(1)
                mask = batch["mask"].to(device).squeeze(1)

                recon = G(inp, k, mask)

                recon_np = recon.cpu().numpy()
                tgt_np = tgt.cpu().numpy()

                for i in range(recon_np.shape[0]):

                    pred = np.squeeze(recon_np[i])
                    gt = np.squeeze(tgt_np[i])

                    pred = pred / (pred.max() + 1e-8)
                    gt = gt / (gt.max() + 1e-8)

                    psnr_scores.append(
                        compare_psnr(gt, pred, data_range=1)
                    )

                    ssim_scores.append(
                        ssim(gt, pred, data_range=1)
                    )

        val_psnr = np.mean(psnr_scores)
        val_ssim = np.mean(ssim_scores)

        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # ---------------------
        # Save Best Model
        # ---------------------
        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                G.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/dagan/best_generator.pt"
            )
        # Save checkpoint each epoch
        if((epoch+1)%10==0):
            torch.save(
                G.state_dict(),
                f"/datasets/tir-datasets-mri-recon/results/checkpoints/dagan/epoch_{epoch+1}.pt"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"D Loss: {epoch_d_loss:.4f} | "
            f"G Loss: {epoch_g_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # ---------------------
    # Save Training History
    # ---------------------
    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/dagan/history.json", "w") as f:
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