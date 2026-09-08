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

def get_noise_schedule(T, device):

    beta = torch.linspace(1e-4, 0.02, T, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return beta, alpha, alpha_bar

class DiffusionUNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1)
        )

    def forward(self, x, t):

        # Append time embedding as extra channel
        t_embed = (t.view(-1,1,1,1) / 1000.0).expand_as(x)
        x = torch.cat([x, t_embed], dim=1)

        return self.net(x)
    
def forward_diffusion(x0, t, alpha_bar):

    noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alpha_bar[t])
    sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar[t])

    xt = sqrt_ab * x0 + sqrt_one_minus_ab * noise

    return xt, noise

def data_consistency(x, k_under, mask):

    x_k = torch.fft.fft2(x.squeeze(1), norm='ortho')
    x_k = mask * k_under + (1 - mask) * x_k
    x_img = torch.fft.ifft2(x_k, norm='ortho')

    return x_img.real.unsqueeze(1)

def train_model(train_loader, val_loader, epochs=10, T=100, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionUNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    beta, alpha, alpha_bar = get_noise_schedule(T, device)

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/diffusion_recon", exist_ok=True)

    history = {
        "train_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    best_psnr = 0

    for epoch in range(epochs):

        # -------------------------
        # TRAINING
        # -------------------------
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            tgt = batch['target'].to(device, non_blocking=True)

            B = tgt.size(0)

            t = torch.randint(0, T, (B,), device=device)

            noise = torch.randn_like(tgt)

            sqrt_ab = torch.sqrt(alpha_bar[t]).view(B,1,1,1)
            sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar[t]).view(B,1,1,1)

            xt = sqrt_ab * tgt + sqrt_one_minus_ab * noise

            pred_noise = model(xt, t.float())

            loss = mse(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        history["train_loss"].append(train_loss)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Validation"):

                tgt = batch['target'].to(device)

                B = tgt.size(0)

                t = torch.randint(0, T, (B,), device=device)

                noise = torch.randn_like(tgt)

                sqrt_ab = torch.sqrt(alpha_bar[t]).view(B,1,1,1)
                sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar[t]).view(B,1,1,1)

                xt = sqrt_ab * tgt + sqrt_one_minus_ab * noise

                pred_noise = model(xt, t.float())

                # denoised estimate
                x0_pred = (xt - sqrt_one_minus_ab * pred_noise) / sqrt_ab

                x0_np = x0_pred.cpu().numpy()
                tgt_np = tgt.cpu().numpy()

                for i in range(B):

                    pred = x0_np[i].squeeze()
                    gt   = tgt_np[i].squeeze()

                    psnr_scores.append(
                        compare_psnr(gt, pred,
                                     data_range=gt.max()-gt.min()+1e-8)
                    )

                    ssim_scores.append(
                        ssim(gt, pred, data_range=1)
                    )

        val_psnr = np.mean(psnr_scores)
        val_ssim = np.mean(ssim_scores)

        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # -------------------------
        # SAVE BEST MODEL
        # -------------------------
        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/diffusion_recon/best_model.pt"
            )

        # Save checkpoint each epoch
        if((epoch+1)%10==0):
            torch.save(
                model.state_dict(),
                f"/datasets/tir-datasets-mri-recon/results/checkpoints/diffusion_recon/epoch_{epoch+1}.pt"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"SSIM: {val_ssim:.4f}"
        )

    # -------------------------
    # SAVE HISTORY
    # -------------------------
    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/diffusion_recon/history.json", "w") as f:
        json.dump(history, f)

    return model, alpha, alpha_bar

def ensure_4d(x):
    """
    Converts tensor to (B,C,H,W)
    """
    while x.dim() > 4:
        x = x.squeeze(1)

    if x.dim() == 3:
        x = x.unsqueeze(1)

    return x

def reconstruct_diffusion(model,
                          k_under,
                          mask,
                          alpha,
                          alpha_bar,
                          T=100):

    device = next(model.parameters()).device

    model.eval()

    k_under = k_under.to(device)
    mask = mask.to(device)

    # Zero-filled initialization
    x = torch.randn_like(
        torch.fft.ifft2(k_under, norm='ortho').real
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():

        with torch.cuda.amp.autocast():

            for t in reversed(range(T)):

                t_tensor = torch.full(
                    (1,), t,
                    device=device,
                    dtype=torch.float32
                )

                xt = ensure_4d(x)
                pred_noise = model(xt, t_tensor.float())

                beta_t = 1 - alpha[t]
                alpha_t = alpha[t]
                alpha_bar_t = alpha_bar[t]

                x = (1/torch.sqrt(alpha_t)) * (
                    x - (beta_t/torch.sqrt(1-alpha_bar_t)) * pred_noise
                )

                if t > 0:
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x)

                # Physics data consistency
                x = data_consistency(x, k_under, mask)

    return x


def evaluate_model(model, loader, alpha=None, alpha_bar=None):

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

                recon = reconstruct_diffusion(
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

    return {k: np.mean(v) for k,v in results.items()}