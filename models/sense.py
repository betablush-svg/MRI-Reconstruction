import h5py
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.ndimage import gaussian_laplace
import lpips

def to_complex_kspace(data):
    data = np.asarray(data)

    if np.iscomplexobj(data):
        return data

    if data.ndim == 4 and data.shape[-1] == 2:
        return data[...,0] + 1j*data[...,1]

    raise ValueError(f"Unexpected k-space format {data.shape}")

def ifft2c(kspace):
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.fftshift(kspace))
    )

def rss_combine(coil_imgs):
    return torch.sqrt(torch.sum(torch.abs(coil_imgs)**2, dim=0)).numpy()

def generate_vd_mask(shape, accel=4, center_fraction=0.08):
    H, W = shape
    mask = np.zeros(W)

    num_low = int(W * center_fraction)
    center = W//2
    mask[center-num_low//2:center+num_low//2] = 1

    prob = (W/accel - num_low) / (W - num_low)
    for i in range(W):
        if mask[i] == 0 and np.random.rand() < prob:
            mask[i] = 1

    return np.tile(mask, (H,1))

def estimate_sensitivity_maps(kspace, acs_size=24):
    coils,H,W = kspace.shape

    # extract ACS region
    cx, cy = H//2, W//2
    acs = kspace[:, cx-acs_size//2:cx+acs_size//2,
                    cy-acs_size//2:cy+acs_size//2]

    coil_imgs_acs = ifft2c(torch.tensor(acs))
    coil_imgs_full = ifft2c(torch.tensor(kspace))

    rss = torch.sqrt(torch.sum(torch.abs(coil_imgs_full)**2, dim=0)) + 1e-8
    sens_maps = coil_imgs_full / rss

    return sens_maps

def compute_sense(kspace,mask, accel=4):
    if isinstance(kspace, torch.Tensor):
        kspace = kspace.squeeze().detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().detach().cpu().numpy()
    kspace_under = kspace * mask
    H,W = kspace.shape

    # ----- Sensitivity estimation -----
    sens_maps = estimate_sensitivity_maps(kspace)

    # ----- SENSE reconstruction -----
    coil_imgs_under = ifft2c(kspace_under)
    img_sense = torch.sum(torch.conj(sens_maps) * coil_imgs_under, dim=0)
    img_sense = torch.abs(img_sense).numpy()
    img_sense /= img_sense.max()

    return img_sense

def sense_reconstruct(full_kspace, accel=4):

    kspace = to_complex_kspace(full_kspace)
    coils,H,W = kspace.shape

    # ----- Ground truth -----
    coil_imgs_full = ifft2c(torch.tensor(kspace))
    img_gt = rss_combine(coil_imgs_full)
    img_gt /= img_gt.max()

    # ----- Undersample -----
    center_fraction = 0.08 if accel==4 else 0.04
    mask = generate_vd_mask((H,W), accel, center_fraction)
    mask = torch.tensor(mask).unsqueeze(0)

    kspace_under = torch.tensor(kspace) * mask

    # ----- Sensitivity estimation -----
    start = time.time()
    sens_maps = estimate_sensitivity_maps(kspace)

    # ----- SENSE reconstruction -----
    coil_imgs_under = ifft2c(kspace_under)
    img_sense = torch.sum(torch.conj(sens_maps) * coil_imgs_under, dim=0)
    img_sense = torch.abs(img_sense).numpy()
    img_sense /= img_sense.max()

    recon_time = time.time() - start

    return img_sense, img_gt, recon_time

def nmse(pred, target):
    return np.linalg.norm(pred-target)**2 / np.linalg.norm(target)**2

def hfen(pred, target):
    return np.linalg.norm(
        gaussian_laplace(pred,1.5)-gaussian_laplace(target,1.5)
    ) / np.linalg.norm(gaussian_laplace(target,1.5))

def vif(pred, target):
    return np.var(pred)/(np.var(target)+1e-8)

lpips_model = lpips.LPIPS(net='alex')

def compute_lpips(pred, target):
    pred  = torch.tensor(pred).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
    target= torch.tensor(target).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
    return lpips_model(pred.float(), target.float()).item()

def evaluate_dataset(folder, accel=4):

    files = list(Path(folder).glob("*.h5"))

    results = {"NMSE":[], "PSNR":[], "SSIM":[],
               "HFEN":[], "VIF":[], "LPIPS":[], "TIME":[]}

    for file in tqdm(files):
        with h5py.File(file,"r") as hf:
            kspace_vol = hf["kspace"][:]

        for slice_k in kspace_vol:
            pred, gt, t = sense_reconstruct(slice_k, accel)

            results["NMSE"].append(nmse(pred,gt))
            results["PSNR"].append(compare_psnr(gt,pred,data_range=1))
            results["SSIM"].append(ssim(gt,pred,data_range=1))
            results["HFEN"].append(hfen(pred,gt))
            results["VIF"].append(vif(pred,gt))
            results["LPIPS"].append(compute_lpips(pred,gt))
            results["TIME"].append(t)

    return results