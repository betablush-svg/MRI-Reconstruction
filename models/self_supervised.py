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

def ensure_4d(x):
    if x.dim() == 5:   # [B,1,N,H,W]
        x = x.mean(dim=2)   # average coils
    elif x.dim() == 4 and x.shape[1] > 1:
        x = x[:,0:1,:,:]    # keep first channel
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    return x

def split_mask(mask, split_ratio=0.4):

    mask1 = mask.clone()
    mask2 = torch.zeros_like(mask)

    idx = torch.nonzero(mask)

    num_split = int(len(idx) * split_ratio)

    perm = torch.randperm(len(idx))
    split_idx = idx[perm[:num_split]]

    for i in split_idx:
        mask1[i[0], i[1]] = 0
        mask2[i[0], i[1]] = 1

    return mask1, mask2

class DenoiserCNN(nn.Module):
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
        return x + self.net(x)
    
class DataConsistency(nn.Module):
    def forward(self, x, k_under, mask):

        x_k = torch.fft.fft2(x, norm='ortho')
        x_k = mask * k_under + (1 - mask) * x_k
        x_img = torch.fft.ifft2(x_k, norm='ortho')

        return x_img.real
    
class SelfSupervisedRecon(nn.Module):
    def __init__(self, iterations=5):
        super().__init__()

        self.iterations = iterations
        self.denoiser = DenoiserCNN()
        self.dc = DataConsistency()

    def forward(self, x, k_under, mask):

        for _ in range(self.iterations):
            x = self.denoiser(x)
            x = self.dc(x, k_under, mask)

        return x
    
   
def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SelfSupervisedRecon().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("/datasets/tir-dataset-mri-recon/results/checkpoints/self_supervised", exist_ok=True)

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    for epoch in range(epochs):

        # -----------------------------
        # TRAINING
        # -----------------------------
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            inp = ensure_4d(batch['img_zf'].to(device, non_blocking=True))
            tgt = ensure_4d(batch['target'].to(device, non_blocking=True))
            k_under = ensure_4d(batch['kspace_under'].to(device, non_blocking=True))
            mask = ensure_4d(batch['mask'].to(device, non_blocking=True))

            B = inp.shape[0]

            optimizer.zero_grad()

            batch_loss = 0

            # SSDU-style mask split per sample
            for i in range(B):

                mask1, mask2 = split_mask(mask[i])

                mask1 = mask1.to(device)
                mask2 = mask2.to(device)

                out = model(inp[i:i+1], k_under[i:i+1], mask1)

                out_k = torch.fft.fft2(out.squeeze(1), norm='ortho')

                loss = torch.mean(torch.abs(mask2 * (out_k - k_under[i]))**2)
                batch_loss += loss

            batch_loss = batch_loss / B

            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        train_loss /= len(train_loader)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()

        val_loss = 0
        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Validation"):

                inp = ensure_4d(batch['img_zf'].to(device))
                tgt = ensure_4d(batch['target'].to(device))
                k_under = ensure_4d(batch['kspace_under'].to(device))
                mask = ensure_4d(batch['mask'].to(device))

                out = model(inp, k_under, mask)

                out_k = torch.fft.fft2(out.squeeze(1), norm='ortho')

                loss = torch.mean(torch.abs(mask * (out_k - k_under))**2)

                val_loss += loss.item()

                out_np = out.cpu().numpy()
                tgt_np = tgt.cpu().numpy()

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

        # -----------------------------
        # SAVE BEST MODEL
        # -----------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-dataset-mri-recon/results/checkpoints/self_supervised/best_model.pt"
            )



        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # -----------------------------
    # SAVE HISTORY
    # -----------------------------
    with open("/datasets/tir-dataset-mri-recon/results/checkpoints/self_supervised/history.json", "w") as f:
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

            inp = ensure_4d(batch['img_zf'].to(device))
            tgt = ensure_4d(batch['target'].to(device))
            k_under = ensure_4d(batch['kspace_under'].to(device))
            mask = ensure_4d(batch['mask'].to(device))

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