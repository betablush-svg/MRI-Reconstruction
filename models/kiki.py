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

class KSpaceBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2,64,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(64,2,3,padding=1)
        )

    def forward(self, k):

        return k + self.net(k)
    
class ImageBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(64,1,3,padding=1)
        )

    def forward(self, x):

        return x + self.net(x)
    
class KIKINet(nn.Module):

    def __init__(self, num_stages=3):

        super().__init__()

        self.num_stages = num_stages

        self.k_blocks = nn.ModuleList(
            [KSpaceBlock() for _ in range(num_stages)]
        )

        self.i_blocks = nn.ModuleList(
            [ImageBlock() for _ in range(num_stages)]
        )


    def forward(self, kspace):

        # kspace shape (B,2,H,W)

        for i in range(self.num_stages):

            # -------------------
            # K-space refinement
            # -------------------

            kspace = self.k_blocks[i](kspace)

            # convert to complex
            k_complex = torch.complex(
                kspace[:,0], kspace[:,1]
            )

            # -------------------
            # IFFT → image
            # -------------------

            img = torch.fft.ifft2(k_complex, norm="ortho").real

            img = img.unsqueeze(1)

            # -------------------
            # Image refinement
            # -------------------

            img = self.i_blocks[i](img)

            # -------------------
            # FFT → k-space
            # -------------------

            k_complex = torch.fft.fft2(img.squeeze(1), norm="ortho")

            kspace = torch.stack(
                [torch.real(k_complex), torch.imag(k_complex)],
                dim=1
            )

        return img
    
   
def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KIKINet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/kiki", exist_ok=True)

    best_val = float("inf")

    history = {"train_loss":[], "val_loss":[]}

    for epoch in range(epochs):

        # -------------------
        # TRAIN
        # -------------------

        model.train()

        train_loss = 0

        for batch in train_loader:
            kspace = batch["kspace_2ch"]
            target = batch["target"]
            kspace = kspace.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            pred = model(kspace)

            loss = criterion(pred,target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------
        # VALIDATION
        # -------------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for batch in val_loader:
                kspace = batch["kspace_2ch"]
                target = batch["target"]
                kspace = kspace.to(device)
                target = target.to(device)

                pred = model(kspace)

                loss = criterion(pred,target)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/kiki/best_model.pt"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_loss:.6f}"
        )

        # ===================================
        # SAVE HISTORY
        # ===================================

        with open("/datasets/tir-datasets-mri-recon/results/checkpoints/kiki/history.json", "w") as f:
            json.dump(history, f)

    return model

def prepare_for_lpips(img):
    img = torch.tensor(img).float()

    # ensure shape is (1,1,H,W)
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)

    # convert to 3-channel
    img = img.repeat(1,3,1,1)

    # normalize to [-1,1]
    img = img * 2 - 1

    return img

def compute_lpips_batch(pred, gt, lpips_model, device):

    # pred, gt shape: (B, 1, H, W)

    # convert to 3-channel
    pred = pred.repeat(1,3,1,1)
    gt   = gt.repeat(1,3,1,1)

    # normalize to [-1,1]
    pred = pred * 2 - 1
    gt   = gt * 2 - 1

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
            inp = batch["kspace_2ch"]
            tgt = batch["target"]
            inp = inp.to(device)
            tgt = tgt.to(device)

            start = time.time()
            out = model(inp)
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

              results["LPIPS"].append(
                  compute_lpips_batch(out, tgt, lpips_model, device)
              )

              results["TIME"].append(recon_time)

    return {k:np.mean(v) for k,v in results.items()}