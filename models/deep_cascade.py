import h5py
import torch
import torch.nn as nn
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

def ensure_4d(x):
    # Always return [B, 1, H, W]
    if x.dim() == 5:
        # If shape is [B, 1, 1, H, W] or [B, 1, N, H, W]
        x = x[:, 0, 0, :, :]   # take first slice/coil
        x = x.unsqueeze(1)     # back to [B, 1, H, W]
    elif x.dim() == 3:
        # If shape is [B, H, W]
        x = x.unsqueeze(1)
    return x

# ================================
# FFT Utilities
# ================================

def fft2c(img):
    return torch.fft.fft2(img, norm="ortho")

def ifft2c(kspace):
    return torch.fft.ifft2(kspace, norm="ortho").real


# ================================
# Data Consistency Layer
# ================================

class DataConsistencyLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img, kspace_under, mask):

        img_k = fft2c(img)

        mask = mask.to(img_k.dtype)

        corrected_k = mask * kspace_under + (1 - mask) * img_k

        img_corrected = ifft2c(corrected_k).real

        return img_corrected


# ================================
# CNN Block
# ================================

class CNNBlock(nn.Module):

    def __init__(self, in_channels=1, features=64, depth=5):

        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_channels, features, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, in_channels, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        res = self.net(x)

        return x + res


# ================================
# Cascade Block
# ================================

class CascadeBlock(nn.Module):

    def __init__(self, in_channels=1):

        super().__init__()

        self.cnn = CNNBlock(in_channels=in_channels)
        self.dc = DataConsistencyLayer()

    def forward(self, img, kspace, mask):

        img = self.cnn(img)

        img = self.dc(img, kspace, mask)

        return img


# ================================
# Deep Cascade CNN
# ================================

class DeepCascadeCNN(nn.Module):

    def __init__(self, num_cascades=5, in_channels=1):

        super().__init__()

        self.cascades = nn.ModuleList(
            [CascadeBlock(in_channels=in_channels) for _ in range(num_cascades)]
        )

    def forward(self, img, kspace, mask):

        x = img

        for cascade in self.cascades:
            x = cascade(x, kspace, mask)

        return x

    

def train_model(train_loader, val_loader, epochs=20, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepCascadeCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("/datasets/tir-dataset-mri-recon/results/checkpoints/deep_cascade", exist_ok=True)

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
            img   = ensure_4d(batch["img_zf"].to(device, non_blocking=True))
            kspace     = ensure_4d(batch["kspace_under"].to(device, non_blocking=True))
            mask  = ensure_4d(batch["mask"].to(device, non_blocking=True))
            target   = ensure_4d(batch["target"].to(device, non_blocking=True))

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
                img   = ensure_4d(batch["img_zf"].to(device))
                kspace     = ensure_4d(batch["kspace_under"].to(device))
                mask  = ensure_4d(batch["mask"].to(device))
                target   = ensure_4d(batch["target"].to(device))

                pred = model(img, kspace, mask)

                loss = criterion(pred, target)

                val_loss += loss.item()

                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()

                for i in range(pred_np.shape[0]):

                    p = pred_np[i,0]
                    g = target_np[i,0]

                    p = p/(p.max()+1e-8)
                    g = g/(g.max()+1e-8)

                    psnr_scores.append(
                        compare_psnr(g, p, data_range=g.max()-g.min()+1e-8)
                    )

                    ssim_scores.append(
                        ssim(g, p, data_range=1)
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
                "/datasets/tir-dataset-mri-recon/results/checkpoints/deep_cascade/best_model.pt"
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

    with open("/datasets/tir-dataset-mri-recon/results/checkpoints/deep_cascade/history.json", "w") as f:
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
            inp   = ensure_4d(batch["img_zf"].to(device))
            kspace = ensure_4d(batch["kspace_under"].to(device))
            mask  = ensure_4d(batch["mask"].to(device))
            tgt   = ensure_4d(batch["target"].to(device))

            start = time.time()
            out = model(inp,kspace,mask)
            recon_time = time.time() - start

            for i in range(out.shape[0]):

              pred = out[i,0].cpu().numpy()
              gt   = tgt[i,0].cpu().numpy()

              #print("pred max:", pred.max(), "gt max:", gt.max())

              pred = pred / (pred.max() + 1e-8)
              gt   = gt   / (gt.max()   + 1e-8)

              nmse_val = np.linalg.norm(pred-gt)**2 / (np.linalg.norm(gt)**2 + 1e-8)

              results["NMSE"].append(
                  nmse_val
              )
              results["PSNR"].append(compare_psnr(gt, pred, data_range=1))
              results["SSIM"].append(ssim(gt,pred,data_range=1))
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