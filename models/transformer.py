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

class PatchEmbedding(nn.Module):

    def __init__(self, img_size=320, patch_size=16, embed_dim=256):

        super().__init__()

        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

      # Ensure tensor is (B,1,H,W)
      if x.dim() == 3:
          x = x.unsqueeze(1)

      elif x.dim() == 5:
          x = x.squeeze(1)

      x = self.proj(x)                 # (B, embed, H/ps, W/ps)

      B, C, H, W = x.shape

      x = x.flatten(2).transpose(1, 2) # (B, N, embed)

      return x, H, W

# -------------------------------
# Transformer Block
# -------------------------------
class TransformerBlock(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4):

        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, x):

        x2 = self.norm1(x)

        attn_out, _ = self.attn(x2, x2, x2)

        x = x + attn_out

        x2 = self.norm2(x)

        x = x + self.mlp(x2)

        return x


# -------------------------------
# Transformer Prior
# -------------------------------
class TransformerPrior(nn.Module):

    def __init__(self,
                 img_size=320,
                 patch_size=16,
                 embed_dim=256,
                 depth=4):

        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)

        self.pos_embed = nn.Parameter(
            torch.randn(1, (img_size // patch_size) ** 2, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim)
            for _ in range(depth)
        ])

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.reconstruct = nn.ConvTranspose2d(
            embed_dim,
            1,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        B = x.shape[0]

        patches, H, W = self.patch_embed(x)

        # --- dynamic positional embedding ---
        pos = self.pos_embed[:, :patches.size(1), :]

        patches = patches + pos

        for blk in self.blocks:
            patches = blk(patches)

        patches = patches.transpose(1, 2).reshape(
            B, self.embed_dim, H, W
        )

        x = self.reconstruct(patches)

        return x


# -------------------------------
# Data Consistency
# -------------------------------
class DataConsistency(nn.Module):

    def forward(self, x, k_under, mask):

        x_k = torch.fft.fft2(x, norm='ortho')

        x_k = mask * k_under + (1 - mask) * x_k

        x_img = torch.fft.ifft2(x_k, norm='ortho')

        return x_img.real


# -------------------------------
# Transformer Reconstruction
# -------------------------------
class TransformerReconstruction(nn.Module):

    def __init__(self, stages=4):

        super().__init__()

        self.stages = stages

        self.prior = TransformerPrior()

        self.dc = DataConsistency()

    def forward(self, x, k_under, mask):

        for _ in range(self.stages):

            x = x + self.prior(x)

            x = self.dc(x, k_under, mask)

        return x
    
def train_model(train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerReconstruction(stages=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs("/datasets/tir-datasets-mri-recon/results/checkpoints/transformer_recon", exist_ok=True)

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

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            inp = ensure_4d(batch['img_zf'].to(device, non_blocking=True).float())
            tgt = ensure_4d(batch['target'].to(device, non_blocking=True).float())
            k_under = ensure_4d(batch['kspace_under'].to(device, non_blocking=True).float())
            mask = ensure_4d(batch['mask'].to(device, non_blocking=True).float())

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
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "/datasets/tir-datasets-mri-recon/results/checkpoints/transformer_recon/best_model.pt"
            )

        # Save checkpoint each epoch
        if((epoch+1)%10==0):
            torch.save(
                model.state_dict(),
                f"/datasets/tir-datasets-mri-recon/results/checkpoints/transformer_recon/epoch_{epoch+1}.pt"
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
    with open("/datasets/tir-datasets-mri-recon/results/checkpoints/transformer_recon/history.json", "w") as f:
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