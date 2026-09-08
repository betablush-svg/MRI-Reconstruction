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
import os, json

def ensure_4d(x):
    """
    Normalize tensor to [B, 1, H, W].
    Handles [B, 1, N, H, W], [B, C, H, W], and [B, H, W].
    """
    if x.dim() == 5:
        # [B, 1, N, H, W] → collapse N
        x = x[:, :, 0, :, :]        # take first slice/coil
        # or: x = x.mean(dim=2)     # average across N slices/coils
    elif x.dim() == 4:
        # [B, C, H, W]
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]     # keep first channel
    elif x.dim() == 3:
        # [B, H, W] → add channel
        x = x.unsqueeze(1)
    return x

class ImageCNN(nn.Module):
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
        return x + self.net(x)  # residual refinement
    
class KspaceCNN(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1)
        )

    def forward(self, k):
        return k + self.net(k)  # residual refinement
    
class DualDomainBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_cnn = ImageCNN()
        self.kspace_cnn = KspaceCNN()

    def forward(self, x, k_under, mask):
        # ----- Image Domain Refinement -----
        x = self.image_cnn(x)  # [B, 1, H, W] ideally

        # If extra dimension exists, collapse it
        if x.dim() == 5:   # [B, 1, N, H, W]
            x = x[:, :, 0, :, :]   # or x.mean(dim=2)

        # ----- FFT -----
        k = torch.fft.fft2(x.squeeze(1), norm='ortho')  # [B, H, W]

        # Convert to 2-channel real representation
        k_real = torch.stack([k.real, k.imag], dim=1)   # [B, 2, H, W]

        # ----- K-space Refinement -----
        k_real = self.kspace_cnn(k_real)                # [B, 2, H, W]

        # Convert back to complex
        k = torch.complex(k_real[:,0], k_real[:,1])     # [B, H, W]

        # ----- Data Consistency -----
        k_under = k_under.squeeze(1)  # [B, H, W]
        mask    = mask.squeeze(1)     # [B, H, W]
        k = mask * k_under + (1 - mask) * k

        # ----- IFFT -----
        x = torch.fft.ifft2(k, norm='ortho').real.unsqueeze(1)  # [B, 1, H, W]

        return x

    
class DualDomainNet(nn.Module):
    def __init__(self, stages=5):
        super().__init__()

        self.stages = nn.ModuleList([
            DualDomainBlock() for _ in range(stages)
        ])

    def forward(self, x, k_under, mask):

        for stage in self.stages:
            x = stage(x, k_under, mask)

        return x

def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualDomainNet(stages=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/dual_domain", exist_ok=True)

    best_val = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    for epoch in range(epochs):

        # -----------------------
        # TRAINING
        # -----------------------
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            inp     = ensure_4d(batch['img_zf'].to(device).float())
            tgt     = ensure_4d(batch['target'].to(device).float())
            k_under = ensure_4d(batch['kspace_under'].to(device).float())
            mask    = ensure_4d(batch['mask'].to(device).float())

            optimizer.zero_grad()

            out = model(inp, k_under, mask)

            loss = criterion(out, tgt)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -----------------------
        # VALIDATION
        # -----------------------
        model.eval()

        val_loss = 0
        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Validation"):

                inp     = ensure_4d(batch['img_zf'].to(device).float())
                tgt     = ensure_4d(batch['target'].to(device).float())
                k_under = ensure_4d(batch['kspace_under'].to(device).float())
                mask    = ensure_4d(batch['mask'].to(device).float())

                out = model(inp, k_under, mask)

                loss = criterion(out, tgt)

                val_loss += loss.item()

                out_np = out.detach().cpu().numpy()
                tgt_np = tgt.detach().cpu().numpy()

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

        val_loss /= len(val_loader)

        val_psnr = np.mean(psnr_scores)
        val_ssim = np.mean(ssim_scores)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # -----------------------
        # SAVE BEST MODEL
        # -----------------------
        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/dual_domain/best_model.pt"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # -----------------------
    # SAVE TRAINING HISTORY
    # -----------------------
    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/dual_domain/history.json", "w") as f:
        json.dump(history, f)

    return model

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

            inp     = ensure_4d(batch['img_zf'].to(device).float())
            tgt     = ensure_4d(batch['target'].to(device).float())
            k_under = ensure_4d(batch['kspace_under'].to(device).float())
            mask    = ensure_4d(batch['mask'].to(device).float())

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