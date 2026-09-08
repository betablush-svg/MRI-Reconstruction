import torch
from utils import dataloader
import time
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from models import diffusion as diff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# LPIPS model (perceptual metric)
lpips_model = lpips.LPIPS(net='alex').to(device)

# ----------------------------
# Metric functions
# ----------------------------
def compute_nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def compute_hfen(gt, pred):
    # Laplacian filter (approximation)
    gt_lap = np.abs(np.gradient(gt)[0]) + np.abs(np.gradient(gt)[1])
    pred_lap = np.abs(np.gradient(pred)[0]) + np.abs(np.gradient(pred)[1])
    return np.linalg.norm(gt_lap - pred_lap)

def compute_lpips(gt, pred):
    gt_t = torch.tensor(gt).unsqueeze(0).unsqueeze(0).float().to(device)
    pred_t = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float().to(device)

    gt_t = gt_t.repeat(1, 3, 1, 1)  # LPIPS needs 3 channels
    pred_t = pred_t.repeat(1, 3, 1, 1)

    return lpips_model(gt_t, pred_t).item()

def compute_vif(gt, pred):
    return np.var(pred)/(np.var(gt)+1e-8)

# ----------------------------
# Normalize helper
# ----------------------------
def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

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
# ----------------------------
# Evaluation function
# ----------------------------
'''def evaluate_model(model, dataloader):
    model.eval()
    
    psnr_list, ssim_list, vif_list = [], [], []
    nmse_list, hfen_list, lpips_list, time_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            
            # Adjust depending on your dataset
            inp   = ensure_4d(batch["img_zf"].to(device))
            kspace = ensure_4d(batch["kspace_under"].to(device))
            mask  = ensure_4d(batch["mask"].to(device))
            gt   = ensure_4d(batch["target"].to(device))

            kspace = batch['kspace_2ch']
            gt = batch['target']
            kspace = kspace.to(device)
            gt = gt.to(device)

            start = time.time()
            #pred = model(inp, kspace, mask)
            #pred = model(kspace)
            pred = model(inp)
            recon_time = time.time() - start

            pred = pred.squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()

            pred = normalize(pred)
            gt = normalize(gt)

            # Metrics
            psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
            ssim = structural_similarity(gt, pred, data_range=1.0)

            nmse = compute_nmse(gt, pred)
            hfen = compute_hfen(gt, pred)
            lp = compute_lpips(gt, pred)
            vif = compute_vif(gt, pred)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            nmse_list.append(nmse)
            hfen_list.append(hfen)
            lpips_list.append(lp)
            vif_list.append(vif)
            time_list.append(recon_time)

    return {
        "PSNR": psnr_list,
        "SSIM": ssim_list,
        "NMSE": nmse_list,
        "HFEN": hfen_list,
        "VIF": vif_list,
        "LPIPS": lpips_list,
        "TIME": time_list
    }'''

def evaluate_model(model, loader, alpha=1, alpha_bar=1):
    psnr_list, ssim_list, vif_list = [], [], []
    nmse_list, hfen_list, lpips_list, time_list = [], [], [], []

    model.eval()

    results = {
        "NMSE":[], "PSNR":[], "SSIM":[],
        "HFEN":[], "VIF":[], "LPIPS":[], "TIME":[]
    }

    device = next(model.parameters()).device
    lpips_model = lpips.LPIPS(net='alex').to(device)

    def to_image(x):
        """Convert tensor to safe 2D numpy image"""
        x = x.detach().cpu().numpy()
        x = np.squeeze(x)

        if x.ndim == 1:
            size = int(np.sqrt(x.size))
            x = x.reshape(size, size)

        return x

    with torch.no_grad():

        for batch in loader:

            inp = batch.get('img_zf', None)
            k_under = batch.get('kspace_under', None)
            mask = batch.get('mask', None)
            tgt = batch['target']

            if inp is not None:
                inp = inp.to(device)

            if k_under is not None:
                k_under = k_under.to(device)

            if mask is not None:
                mask = mask.to(device)

            tgt = tgt.to(device)

            # ensure shape (B,1,H,W)
            if tgt.dim() == 3:
                tgt = tgt.unsqueeze(1)

            start = time.time()

            # diffusion models
            if alpha is not None and alpha_bar is not None:

                recon = diff.reconstruct_diffusion(
                    model,
                    k_under,
                    mask,
                    alpha,
                    alpha_bar
                )

            # normal reconstruction models
            elif inp is not None and k_under is not None and mask is not None:

                recon = model(inp, k_under, mask)

            elif inp is not None:

                recon = model(inp)

            else:
                raise ValueError("Invalid model input configuration.")

            recon_time = time.time() - start

            recon = recon.cpu()

            B = recon.shape[0]

            for i in range(B):

                pred = to_image(recon[i])
                gt = to_image(tgt[i])

                # force identical size
                H = min(pred.shape[-2], gt.shape[-2])
                W = min(pred.shape[-1], gt.shape[-1])

                pred = pred[:H,:W]
                gt = gt[:H,:W]

                pred = pred/(pred.max()+1e-8)
                gt = gt/(gt.max()+1e-8)

                # NMSE
                nmse_val = np.linalg.norm(pred-gt)**2 / (np.linalg.norm(gt)**2+1e-8)

                # PSNR
                psnr_val = compare_psnr(gt, pred, data_range=1)

                # SSIM
                ssim_val = ssim(gt, pred, data_range=1, win_size=7)

                # HFEN
                hfen_val = np.linalg.norm(
                    gaussian_laplace(pred,1.5) -
                    gaussian_laplace(gt,1.5)
                ) / (np.linalg.norm(gaussian_laplace(gt,1.5))+1e-8)

                # VIF (simple proxy)
                vif_val = np.var(pred)/(np.var(gt)+1e-8)

                # LPIPS
                pred_lp = torch.tensor(pred).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float().to(device)
                gt_lp   = torch.tensor(gt).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float().to(device)

                pred_lp = pred_lp*2-1
                gt_lp   = gt_lp*2-1

                lpips_val = lpips_model(pred_lp, gt_lp).mean().item()

                results["NMSE"].append(nmse_val)
                results["PSNR"].append(psnr_val)
                results["SSIM"].append(ssim_val)
                results["HFEN"].append(hfen_val)
                results["VIF"].append(vif_val)
                results["LPIPS"].append(lpips_val)
                results["TIME"].append(recon_time)

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                nmse_list.append(nmse)
                hfen_list.append(hfen)
                lpips_list.append(lp)
                vif_list.append(vif)
                time_list.append(recon_time)
    return {
        "PSNR": psnr_list,
        "SSIM": ssim_list,
        "NMSE": nmse_list,
        "HFEN": hfen_list,
        "VIF": vif_list,
        "LPIPS": lpips_list,
        "TIME": time_list
    }

# ----------------------------
# Load model (modify if needed)
# ----------------------------
def load_model(model_path,input_shape):
    model = diff.DiffusionUNet() # initialize your model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----------------------------
# RUN
# ----------------------------

undersampled_path = "datasets\\knee_multicoil_undersampled_x4"

train_set = dataloader.MRIDataset(undersampled_path)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=1)
model_path = "checkpoints\\ALL_MODELS\\DIFFUSION_best_model_x4.pt"
model_name = "DIFFUSION_x4"

ckpt = torch.load(model_path, map_location=device)
print(type(ckpt))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = next(iter(train_loader))
input_shape = batch['kspace_2ch'].shape[1:]

model = load_model(model_path,input_shape)
metrics = evaluate_model(model, train_loader)
df = pd.DataFrame(metrics)
print(df)
# Save
df.to_csv(f"{model_name}.csv", index=False)

print("\nSaved results to CSV & Excel")