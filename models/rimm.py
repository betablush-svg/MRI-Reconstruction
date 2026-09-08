import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def data_consistency_grad(x, kspace_under, mask):

    # x shape (B,1,H,W)

    x_k = torch.fft.fft2(x.squeeze(1), norm="ortho")

    diff = mask.squeeze(1) * (x_k - kspace_under.squeeze(1))

    grad = torch.fft.ifft2(diff, norm="ortho").real

    return grad.unsqueeze(1)

class RIMBlock(nn.Module):

    def __init__(self, hidden_channels=64):

        super().__init__()

        self.conv_in = nn.Conv2d(2, hidden_channels, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gru = nn.GRUCell(hidden_channels, hidden_channels)

        self.conv_out = nn.Conv2d(hidden_channels, 1, 3, padding=1)


    def forward(self, x, grad, h):

        inp = torch.cat([x, grad], dim=1)

        feat = torch.relu(self.conv_in(inp))     # (B,64,H,W)

        pooled = self.pool(feat).view(feat.size(0), -1)   # (B,64)

        h = self.gru(pooled, h)                  # GRU update

        h_map = h.view(h.size(0), h.size(1), 1, 1)

        feat = feat + h_map                      # broadcast update

        dx = self.conv_out(feat)

        x = x + dx

        return x, h
    
class RIM(nn.Module):

    def __init__(self, steps=8, hidden_channels=64):

        super().__init__()

        self.steps = steps

        self.block = RIMBlock(hidden_channels)


    def forward(self, img_zf, kspace_under, mask):

        x = img_zf

        B = x.shape[0]

        h = torch.zeros(B, 64, device=x.device)

        for _ in range(self.steps):

            grad = data_consistency_grad(x, kspace_under, mask)

            x, h = self.block(x, grad, h)

        return x
    
def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RIM(steps=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/rim", exist_ok=True)

    best_val = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    for epoch in range(epochs):

        # -------------------------
        # TRAINING
        # -------------------------
        model.train()

        train_loss = 0

        for batch in tqdm(train_loader,
                                            desc=f"Epoch {epoch+1} Training"):

            inp = batch['img_zf'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)
            k_under = batch['kspace_under'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            optimizer.zero_grad()

            out = model(inp, k_under, mask)

            loss = criterion(out, tgt)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        val_loss = 0
        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in tqdm(val_loader,
                                                desc="Validation"):

                inp = batch['img_zf'].to(device)
                tgt = batch['target'].to(device)
                k_under = batch['kspace_under'].to(device)
                mask = batch['mask'].to(device)

                out = model(inp, k_under, mask)

                loss = criterion(out, tgt)

                val_loss += loss.item()

                out_np = out.detach().cpu().numpy()
                tgt_np = tgt.detach().cpu().numpy()

                for i in range(out_np.shape[0]):

                    pred = out_np[i].squeeze()
                    gt   = tgt_np[i].squeeze()

                    psnr_scores.append(
                        compare_psnr(gt, pred,
                                     data_range=gt.max()-gt.min()+1e-8)
                    )

                    ssim_scores.append(
                        ssim(gt, pred, data_range=1)
                    )

        val_loss /= len(val_loader)

        val_psnr = np.mean(psnr_scores)
        val_ssim = np.mean(ssim_scores)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # -------------------------
        # SAVE BEST MODEL
        # -------------------------
        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/rim/best_model.pt"
            )


        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # -------------------------
    # SAVE TRAINING HISTORY
    # -------------------------
    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/rim/history.json", "w") as f:
        json.dump(history, f)

    return model

def prepare_for_lpips(img):

    img = torch.tensor(img).float()

    # ensure shape (1,1,H,W)
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)

    elif img.ndim == 3:
        img = img.unsqueeze(0)

    # convert to 3 channels
    img = img.repeat(1,3,1,1)

    # normalize [0,1] → [-1,1]
    img = img * 2 - 1

    return img

def compute_lpips_batch(pred, gt, lpips_model, device):

    # pred, gt shape (B,1,H,W)

    pred = pred.repeat(1,3,1,1)
    gt   = gt.repeat(1,3,1,1)

    pred = pred * 2 - 1
    gt   = gt * 2 - 1

    pred = pred.to(device)
    gt   = gt.to(device)

    return lpips_model(pred, gt).mean().item()

def evaluate_model(model, loader):    
    device = next(model.parameters()).device
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    results = {"NMSE":[], "PSNR":[], "SSIM":[],
               "HFEN":[], "VIF":[], "LPIPS":[], "TIME":[]}

    model.eval()

    with torch.no_grad():
        for batch in loader:
            inp = batch['img_zf'].to(device)
            tgt = batch['target'].to(device)
            k_under = batch['kspace_under'].to(device)
            mask = batch['mask'].to(device)

            start = time.time()
            out = model(inp, k_under, mask)
            recon_time = time.time() - start

            for i in range(out.shape[0]):

              pred = out[i,0].squeeze().cpu().numpy()
              gt   = tgt[i,0].squeeze().cpu().numpy()

              print("pred max:", pred.max(), "gt max:", gt.max())

              pred = pred / (pred.max() + 1e-8)
              gt   = gt   / (gt.max()   + 1e-8)

              den = np.linalg.norm(gt)**2
              if den == 0:
                continue
              nmse_val = np.linalg.norm(pred-gt)**2 / (den + 1e-8)

              results["NMSE"].append(
                  nmse_val
              )
              results["PSNR"].append(compare_psnr(gt, pred, data_range=gt.max() - gt.min() + 1e-8))
              results["SSIM"].append(ssim(gt,pred,data_range=1, win_size=7))
              results["HFEN"].append(
                  np.linalg.norm(
                      gaussian_laplace(pred,1.5) -
                      gaussian_laplace(gt,1.5)
                  )/np.linalg.norm(gaussian_laplace(gt,1.5))
              )
              results["VIF"].append(np.var(pred)/(np.var(gt)+1e-8))

              pred_lp = prepare_for_lpips(pred)
              gt_lp   = prepare_for_lpips(gt)

              results["LPIPS"].append(compute_lpips_batch(out[i:i+1], tgt[i:i+1], lpips_model, device))

              results["TIME"].append(recon_time)

    return {k:np.mean(v) for k,v in results.items()}