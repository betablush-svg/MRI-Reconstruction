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
    # Always return [B, 1, H, W]
    if x.dim() == 5:
        # If shape is [B, 1, N, H, W], take the first slice/coil
        x = x[:, :, 0, :, :]   # → [B, 1, H, W]
    elif x.dim() == 3:
        # If shape is [B, H, W], add channel
        x = x.unsqueeze(1)
    return x

# -------------------------------------------------
# Window Attention
# -------------------------------------------------
class WindowAttention(nn.Module):

    def __init__(self, dim, num_heads):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True
        )

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)

        return attn_out


# -------------------------------------------------
# Swin Block
# -------------------------------------------------
class SwinBlock(nn.Module):

    def __init__(self, dim=48, num_heads=4, window_size=4):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        self.window_size = window_size
        self.dim = dim


    # ----------------------------
    # Window Partition
    # ----------------------------
    def window_partition(self, x):

        B, H, W, C = x.shape
        ws = self.window_size

        x = x.view(
            B,
            H//ws, ws,
            W//ws, ws,
            C
        )

        windows = x.permute(0,1,3,2,4,5).contiguous()

        windows = windows.view(-1, ws*ws, C)

        return windows


    # ----------------------------
    # Window Reverse
    # ----------------------------
    def window_reverse(self, windows, H, W):

        ws = self.window_size

        B = int(windows.shape[0] / ((H/ws)*(W/ws)))

        x = windows.view(
            B,
            H//ws, W//ws,
            ws, ws,
            self.dim
        )

        x = x.permute(0,1,3,2,4,5).contiguous()

        x = x.view(B, H, W, self.dim)

        return x


    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x):

        # -------- shape guard --------
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.dim() == 5:
            x = x.squeeze(1)

        B, C, H, W = x.shape

        ws = self.window_size

        # ensure divisibility
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0,pad_w,0,pad_h))

        B, C, H_pad, W_pad = x.shape

        x = x.permute(0,2,3,1)   # (B,H,W,C)

        windows = self.window_partition(x)

        # attention
        attn = self.attn(self.norm1(windows))

        windows = windows + attn

        # MLP
        windows = windows + self.mlp(self.norm2(windows))

        x = self.window_reverse(windows, H_pad, W_pad)

        x = x.permute(0,3,1,2)

        # remove padding
        x = x[:, :, :H, :W]

        return x


# -------------------------------------------------
# Swin Prior
# -------------------------------------------------
class SwinPrior(nn.Module):
    def __init__(self, dim=48, depth=2):
        super().__init__()
        self.embed = nn.Conv2d(1, dim, 3, padding=1)
        self.blocks = nn.Sequential(*[SwinBlock(dim=dim) for _ in range(depth)])
        self.recon = nn.Conv2d(dim, 1, 3, padding=1)


    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.dim() == 5:
            x = x.squeeze(1)

        x_embed = self.embed(x)

        x_embed = self.blocks(x_embed)

        out = self.recon(x_embed)

        return out


# -------------------------------------------------
# Data Consistency
# -------------------------------------------------
class DataConsistency(nn.Module):
    def forward(self, x, k_under, mask):
        x_k = torch.fft.fft2(x, norm='ortho')
        x_k = mask * k_under + (1 - mask) * x_k
        x_img = torch.fft.ifft2(x_k, norm='ortho').real
        return x_img



# -------------------------------------------------
# Full Reconstruction Network
# -------------------------------------------------
class SwinReconstruction(nn.Module):

    def __init__(self, stages=4):

        super().__init__()

        self.stages = stages

        self.prior = SwinPrior()

        self.dc = DataConsistency()


    def forward(self, x, k_under, mask):

        for _ in range(self.stages):

            x = x + self.prior(x)

            x = self.dc(x, k_under, mask)

        return x
    

def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinReconstruction(stages=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/swin_recon", exist_ok=True)

    best_val_loss = float("inf")

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
        scaler = torch.cuda.amp.GradScaler()

        for batch in tqdm(train_loader,
                                            desc=f"Epoch {epoch+1} Training"):

            inp = ensure_4d(batch['img_zf'].to(device, non_blocking=True).float())
            tgt = ensure_4d(batch['target'].to(device, non_blocking=True).float())
            k_under = ensure_4d(batch['kspace_under'].to(device, non_blocking=True).float())
            mask = ensure_4d(batch['mask'].to(device, non_blocking=True).float())

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                

                out = model(inp, k_under, mask)

                loss = criterion(out, tgt)

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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

            for batch in tqdm(val_loader,
                                                desc="Validation"):

                inp = ensure_4d(batch['img_zf'].to(device).float())
                tgt = ensure_4d(batch['target'].to(device).float())
                k_under = ensure_4d(batch['kspace_under'].to(device).float())
                mask = ensure_4d(batch['mask'].to(device).float())

                out = model(inp, k_under, mask)

                loss = criterion(out, tgt)

                val_loss += loss.item()

                out_np = out.detach().cpu().numpy()
                tgt_np = tgt.detach().cpu().numpy()

                for i in range(out_np.shape[0]):

                    pred = out_np[i].squeeze()
                    gt   = tgt_np[i].squeeze()

                    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    gt   = (gt   - gt.min()) / (gt.max() - gt.min() + 1e-8)


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
        torch.cuda.empty_cache()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # -----------------------
        # SAVE BEST MODEL
        # -----------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/swin_recon/best_model.pt"
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
    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/swin_recon/history.json", "w") as f:
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

            inp = ensure_4d(batch['img_zf'].to(device).float())
            tgt = ensure_4d(batch['target'].to(device).float())
            k_under = ensure_4d(batch['kspace_under'].to(device).float())
            mask = ensure_4d(batch['mask'].to(device).float())

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