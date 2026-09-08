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

  
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
    
def train_model(train_loader, val_loader, epochs=20, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/srcnn", exist_ok=True)

    best_val_loss = float("inf")

    # -----------------------------
    # Metric history
    # -----------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    for epoch in range(epochs):

        # =============================
        # TRAINING
        # =============================

        model.train()

        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            inp = batch["img_zf"]
            tgt = batch["target"]

            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad()

            out = model(inp)

            loss = criterion(out, tgt)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # =============================
        # VALIDATION
        # =============================

        model.eval()

        val_loss = 0
        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Validation"):

                inp = batch["img_zf"]
                tgt = batch["target"]
                inp = inp.to(device)
                tgt = tgt.to(device)

                out = model(inp)

                loss = criterion(out, tgt)

                val_loss += loss.item()

                out_np = out.cpu().numpy()
                tgt_np = tgt.cpu().numpy()

                for i in range(out_np.shape[0]):

                    pred = out_np[i,0]
                    gt   = tgt_np[i,0]

                    psnr_scores.append(
                        compare_psnr(gt, pred, data_range=gt.max()-gt.min()+1e-8)
                    )

                    ssim_scores.append(
                        ssim(gt, pred, data_range=1)
                    )

        val_loss /= len(val_loader)

        val_psnr = np.mean(psnr_scores)
        val_ssim = np.mean(ssim_scores)

        # =============================
        # STORE HISTORY
        # =============================

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # =============================
        # SAVE BEST MODEL
        # =============================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/srcnn/best_model.pt"
            )



        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # =============================
    # SAVE HISTORY FOR FUTURE USE
    # =============================

    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/srcnn/history.json", "w") as f:
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
            inp = batch["img_zf"]
            tgt = batch["target"]
            inp = inp.to(device)
            tgt = tgt.to(device)

            start = time.time()
            out = model(inp)
            recon_time = time.time() - start

            for i in range(out.shape[0]):

                pred = out[i,0].cpu().numpy()
                gt   = tgt[i,0].cpu().numpy()

                pred = pred / (pred.max() + 1e-8)
                gt   = gt   / (gt.max() + 1e-8)

                nmse = np.linalg.norm(pred-gt)**2 / (np.linalg.norm(gt)**2 + 1e-8)

                results["NMSE"].append(nmse)
                results["PSNR"].append(compare_psnr(gt,pred,data_range=1))
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

                results["LPIPS"].append(compute_lpips_batch(out, tgt, lpips_model, device))
                results["TIME"].append(recon_time)

    return {k:np.mean(v) for k,v in results.items()}