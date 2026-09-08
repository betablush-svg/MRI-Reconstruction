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
import lpips
import os
import json

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, base=64):

        super().__init__()

        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)

        self.pool = nn.MaxPool2d(2)

        self.middle = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.final = nn.Conv2d(base, out_ch, 1)

    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        m = self.middle(self.pool(e4))

        d4 = self.up4(m)
        d4 = self.dec4(torch.cat([d4,e4],dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3,e3],dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2,e2],dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1,e1],dim=1))

        return self.final(d1)
    
class DataConsistency(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, k_under, mask):

        # x: (B,1,H,W)

        x_k = torch.fft.fft2(x, norm="ortho")

        mask = mask.to(x_k.dtype)

        corrected_k = mask * k_under + (1 - mask) * x_k

        x_img = torch.fft.ifft2(corrected_k, norm="ortho").real

        return x_img
    
class PhysicsUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.unet = UNet(in_ch=1,out_ch=1)
        self.dc = DataConsistency()

    def forward(self,x,k_under,mask):

        x = self.unet(x)

        x = self.dc(x,k_under,mask)

        return x

 
def train_model(train_loader, val_loader, epochs=20, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhysicsUNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/unetx8", exist_ok=True)

    best_val_loss = float("inf")

    # --------------------------------
    # History dictionary
    # --------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    for epoch in range(epochs):

        # ===================================
        # TRAINING
        # ===================================

        model.train()

        train_loss = 0

        for batch in tqdm(train_loader,
                                              desc=f"Epoch {epoch+1} Training"):
            img   = batch["img_zf"].to(device)
            kspace     = batch["kspace_under"].to(device)
            mask  = batch["mask"].to(device)
            target   = batch["target"].to(device)

            optimizer.zero_grad()

            pred = model(img, kspace, mask)

            loss = criterion(pred, target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===================================
        # VALIDATION
        # ===================================

        model.eval()

        val_loss = 0
        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in val_loader:

                inp = batch["img_zf"].to(device)
                tgt = batch["target"].to(device)
                k_under = batch["kspace_under"].to(device)
                mask = batch["mask"].to(device)

                out = model(inp, k_under, mask)

                loss = criterion(out, tgt)  # reduce extra dimension
                val_loss += loss.item()

                out_np = out.cpu().numpy()
                tgt_np = tgt.cpu().numpy()

                for i in range(out_np.shape[0]):

                    # out[i] = [1,4,H,W]
                    pred = out_np[i,0]          # [4,H,W]
                    gt = tgt_np[i,0]

                    pred = pred/(pred.max()+1e-8)
                    gt = gt/(gt.max()+1e-8)

                    psnr_scores.append(
                        compare_psnr(gt, pred, data_range=1)
                    )

                    ssim_scores.append(
                        ssim(gt, pred, data_range=1)
                    )

        val_loss /= len(val_loader)

        val_psnr = np.mean(psnr_scores)
        val_ssim = np.mean(ssim_scores)

        # ===================================
        # STORE HISTORY
        # ===================================

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # ===================================
        # SAVE BEST MODEL
        # ===================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/unetx8/best_model.pt"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # ===================================
    # SAVE HISTORY
    # ===================================

    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/unetx8/history.json", "w") as f:
        json.dump(history, f)

    return model

def evaluate_model(model, loader):

    device = next(model.parameters()).device
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    results = {
        "NMSE":[], "PSNR":[], "SSIM":[],
        "HFEN":[], "VIF":[], "LPIPS":[], "TIME":[]
    }

    model.eval()

    with torch.no_grad():
        for batch in loader:
            inp   = batch["img_zf"].to(device)
            k_under    = batch["kspace_under"].to(device)
            mask  = batch["mask"].to(device)
            tgt   = batch["target"].to(device)

            start = time.time()
            out = model(inp, k_under, mask)
            recon_time = time.time() - start

            for i in range(out.shape[0]):

                pred = out[i,0].cpu().numpy()
                gt   = tgt[i,0].cpu().numpy()

                pred = pred / (pred.max() + 1e-8)
                gt   = gt   / (gt.max() + 1e-8)

                nmse_val = np.linalg.norm(pred-gt)**2 / (np.linalg.norm(gt)**2 + 1e-8)

                results["NMSE"].append(nmse_val)

                results["PSNR"].append(
                    compare_psnr(gt, pred, data_range=1)
                )

                results["SSIM"].append(
                    ssim(gt, pred, data_range=1)
                )

                results["HFEN"].append(
                    np.linalg.norm(
                        gaussian_laplace(pred,1.5) -
                        gaussian_laplace(gt,1.5)
                    ) / (np.linalg.norm(gaussian_laplace(gt,1.5)) + 1e-8)
                )

                results["VIF"].append(
                    np.var(pred)/(np.var(gt)+1e-8)
                )

                # LPIPS
                pred_lp = pred[None,None,:,:]
                gt_lp   = gt[None,None,:,:]

                pred_lp = torch.tensor(pred_lp).repeat(1,3,1,1).float().to(device)
                gt_lp   = torch.tensor(gt_lp).repeat(1,3,1,1).float().to(device)

                pred_lp = pred_lp * 2 - 1
                gt_lp   = gt_lp * 2 - 1

                lpips_val = lpips_model(pred_lp, gt_lp).mean().item()
                results["LPIPS"].append(lpips_val)

                results["TIME"].append(recon_time)

    return {k: np.mean(v) for k,v in results.items()}