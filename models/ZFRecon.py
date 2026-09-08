import h5py
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.ndimage import gaussian_laplace
import lpips
from sewar.full_ref import vifp

def to_tensor(data):
    data = torch.tensor(data)
    if torch.is_complex(data):
        return data
    if data.ndim >= 3 and data.shape[-1] == 2:
        return torch.view_as_complex(data.float())
    return data

def ifft2c(kspace):
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.fftshift(kspace))
    )

def rss_combine(coil_imgs):
    return torch.sqrt(torch.sum(torch.abs(coil_imgs)**2, dim=0))


## Metrics

def nmse(pred, target):
    return np.linalg.norm(pred-target)**2 / np.linalg.norm(target)**2

def hfen(pred, target, sigma=1.5):
    log_pred = gaussian_laplace(pred, sigma=sigma)
    log_gt   = gaussian_laplace(target, sigma=sigma)

    num = np.linalg.norm(log_pred - log_gt)
    den = np.linalg.norm(log_gt) + 1e-8  # avoid division by zero

    return num / den

def vif(pred, target):
    return vifp(target, pred)  # NOTE: order is (ref, dist)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()
def compute_lpips(pred, target):


    # Convert to tensor
    pred  = torch.tensor(pred, dtype=torch.float32).to(device)
    target= torch.tensor(target, dtype=torch.float32).to(device)

    # Shape: [1, 1, H, W]
    pred  = pred.unsqueeze(0).unsqueeze(0)
    target= target.unsqueeze(0).unsqueeze(0)

    # Repeat to 3 channels
    pred  = pred.repeat(1, 3, 1, 1)
    target= target.repeat(1, 3, 1, 1)

    # Normalize to [-1, 1]
    pred  = pred * 2 - 1
    target= target * 2 - 1

    with torch.no_grad():
        val = lpips_model(pred, target)

    return val.item()

def reconstruct_from_kspace(kspace):
    kspace = to_tensor(kspace)
    if kspace.ndim == 2:
        kspace = kspace.unsqueeze(0)

    coil_imgs = ifft2c(kspace)
    img = rss_combine(coil_imgs).numpy()
    img /= img.max()
    return img


def evaluate_image_dataset(dataloader):

    #files = list(Path(folder).glob("*.h5"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    results = {"NMSE":[], "PSNR":[], "SSIM":[],
               "HFEN":[], "VIF":[], "LPIPS":[], "TIME":[]}

    for batch in dataloader:

        ks = batch["kspace_under"]
        m   = batch["mask"]
        inputs  = batch["img_zf"]  # ZF images
        gt = batch["target"]  # Fully sampled images
            
        start = time.time()
        masked_kspace = ks * m
        pred = reconstruct_from_kspace(masked_kspace)
        end = time.time() - start
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        # Convert to numpy safely
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        # Ensure float type
        pred = pred.astype(np.float32)
        gt   = gt.astype(np.float32)

        scale = gt.max()
        pred = pred / scale
        gt   = gt / scale
        

        results["NMSE"].append(nmse(pred, gt))
        results["PSNR"].append(compare_psnr(gt, pred, data_range=1))
        results["SSIM"].append(ssim(gt, pred, data_range=1))
        results["HFEN"].append(hfen(pred, gt))
        results["VIF"].append(vif(pred, gt))
        results["LPIPS"].append(compute_lpips(pred, gt))
        results["TIME"].append(end)


    return results
