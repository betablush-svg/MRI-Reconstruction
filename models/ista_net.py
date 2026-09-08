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
    # Always return [B, 1, H, W]
    if x.dim() == 5:
        # If shape is [B, 1, 1, H, W] or [B, 1, N, H, W]
        x = x[:, 0, 0, :, :]   # take first slice/coil
        x = x.unsqueeze(1)     # back to [B, 1, H, W]
    elif x.dim() == 3:
        # If shape is [B, H, W]
        x = x.unsqueeze(1)
    return x

class ISTABlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.lambda_step = nn.Parameter(torch.tensor(0.5))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x, k_under, mask):

        # -------- Data Consistency Gradient Step --------
        x_k = torch.fft.fft2(x, norm='ortho')
        grad = torch.fft.ifft2(mask * (x_k - k_under), norm='ortho').real
        #grad = grad.unsqueeze(1)

        x = x - self.lambda_step * grad

        # -------- Learned Proximal Mapping --------
        z = self.conv1(x)
        z = self.relu(z)
        z = self.conv2(z)

        # Soft thresholding
        x = torch.sign(z) * torch.relu(torch.abs(z) - self.soft_thr)

        return x
    
class ISTANet(nn.Module):
    def __init__(self, phases=8):
        super().__init__()

        self.phases = phases
        self.blocks = nn.ModuleList([
            ISTABlock() for _ in range(phases)
        ])

    def forward(self, x, k_under, mask):

        for i in range(self.phases):
            x = self.blocks[i](x, k_under, mask)

        return x
    

def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ISTANet(phases=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/istanet", exist_ok=True)

    best_val_loss = float("inf")

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

            inp = ensure_4d(batch['img_zf'].to(device, non_blocking=True).float())
            tgt = ensure_4d(batch['target'].to(device, non_blocking=True).float())
            k_under = ensure_4d(batch['kspace_under'].to(device, non_blocking=True).float())
            mask = ensure_4d(batch['mask'].to(device, non_blocking=True).float())
            # If input accidentally has 5D, squeeze the extra dimension
            if inp.dim() == 5:
                # Example: [B, 1, 1, H, W] → [B, 1, H, W]
                inp = inp.squeeze(2)
            if tgt.dim() == 5:
                tgt = tgt.squeeze(2)
            if k_under.dim() == 5:
                k_under = k_under.squeeze(2)
            if mask.dim() == 5:
                mask = mask.squeeze(2)           

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

                inp = ensure_4d(batch['img_zf'].to(device).float())
                tgt = ensure_4d(batch['target'].to(device).float())
                k_under = ensure_4d(batch['kspace_under'].to(device).float())
                mask = ensure_4d(batch['mask'].to(device).float())
              

                out = model(inp, k_under, mask)

                loss = criterion(out, tgt)

                val_loss += loss.item()

                out_np = out.cpu().numpy()
                tgt_np = tgt.cpu().numpy()

                for i in range(out_np.shape[0]):
                    pred = out_np[i].squeeze()
                    gt   = tgt_np[i].squeeze()

                    psnr_scores.append(
                        compare_psnr(gt, pred, data_range=gt.max()-gt.min()+1e-8)
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
        # Save best model
        # -------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/istanet/best_model.pt"
            )


        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # -------------------------
    # Save training history
    # -------------------------

    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/istanet/history.json", "w") as f:
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

            inp = ensure_4d(batch['img_zf'].to(device).float())
            tgt = ensure_4d(batch['target'].to(device).float())
            k_under = ensure_4d(batch['kspace_under'].to(device).float())
            mask = ensure_4d(batch['mask'].to(device).float())
            if inp.dim() == 5:
                inp = inp.squeeze(2)

            if tgt.dim() == 5:
                tgt = tgt.squeeze(2)

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

                # LPIPS
                pred_lp = torch.tensor(pred[None,None,:,:]).repeat(1,3,1,1).float().to(device)
                gt_lp   = torch.tensor(gt[None,None,:,:]).repeat(1,3,1,1).float().to(device)

                pred_lp = pred_lp * 2 - 1
                gt_lp   = gt_lp * 2 - 1

                lpips_val = lpips_model(pred_lp, gt_lp).mean().item()
                results["LPIPS"].append(lpips_val)

                results["TIME"].append(recon_time)

    return {k: np.mean(v) for k,v in results.items()}