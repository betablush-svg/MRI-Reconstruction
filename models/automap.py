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

class AUTOMAP(nn.Module):

    def __init__(self, input_shape):

        super().__init__()

        C, H, W = input_shape
        self.H = H
        self.W = W

        input_dim = C * H * W
        output_dim = H * W

        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, output_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(1,64,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(64,64,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(64,1,5,padding=2)
        )

    def forward(self, kspace):

        B = kspace.shape[0]

        x = kspace.view(B,-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = x.view(B,1,self.H,self.W)

        x = self.conv(x)

        return x
    
   
def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = next(iter(train_loader))
    input_shape = batch['kspace_2ch'].shape[1:]   # (C,H,W)

    model = AUTOMAP(input_shape).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/automap", exist_ok=True)

    best_val = float("inf")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):

        # -------------------------
        # TRAIN
        # -------------------------

        model.train()

        train_loss = 0

        for batch in train_loader:
            kspace = batch['kspace_2ch']
            target = batch['target']
            kspace = kspace.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            pred = model(kspace)

            loss = criterion(pred, target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for batch in val_loader:
                kspace = batch['kspace_2ch']
                target = batch['target']
                kspace = kspace.to(device)
                target = target.to(device)

                pred = model(kspace)

                loss = criterion(pred, target)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # save best model
        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/automap/best_model.pt"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_loss:.6f}"
        )

        # ===================================
        # SAVE HISTORY
        # ===================================

        with open("/datasets/tir-datasets-mri-recon/results/checkpoints/automap/history.json", "w") as f:
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
            inp = batch['kspace_2ch']
            tgt = batch['target']
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

              results["LPIPS"].append(compute_lpips_batch(out[i:i+1], tgt[i:i+1], lpips_model, device))

              results["TIME"].append(recon_time)

    return {k:np.mean(v) for k,v in results.items()}