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
    raise ValueError("Unexpected k-space format")

def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace)))

def rss_combine(coil_imgs):
    return np.sqrt(np.sum(np.abs(coil_imgs)**2, axis=0))

def generate_vd_mask(shape, accel=4, center_fraction=0.08):
    H,W = shape
    mask = np.zeros(W)

    num_low = int(W*center_fraction)
    center = W//2
    mask[center-num_low//2:center+num_low//2] = 1

    prob = (W/accel - num_low)/(W-num_low)
    for i in range(W):
        if mask[i]==0 and np.random.rand()<prob:
            mask[i]=1

    return np.tile(mask,(H,1))

def lowrank_shrinkage(kspace, lam=0.02):

    coils,H,W = kspace.shape

    # reshape to 2D matrix (coil stacking)
    mat = kspace.reshape(coils, H*W)

    # SVD
    U,S,Vh = np.linalg.svd(mat, full_matrices=False)

    # Soft threshold singular values
    S_thresh = np.maximum(S - lam, 0)

    mat_lowrank = (U * S_thresh[:,None]) @ Vh

    return mat_lowrank.reshape(coils,H,W)

def lowrank_pi_recon(full_kspace,
                     accel=4,
                     n_iter=20,
                     lam=0.02):

    kspace = to_complex_kspace(full_kspace)
    coils,H,W = kspace.shape

    # ----- Ground truth -----
    coil_imgs_full = ifft2c(kspace)
    img_gt = rss_combine(coil_imgs_full)
    img_gt /= img_gt.max()

    # ----- Undersample -----
    center_fraction = 0.08 if accel==4 else 0.04
    mask = generate_vd_mask((H,W), accel, center_fraction)
    mask = mask[np.newaxis,:,:]

    kspace_under = kspace * mask
    kspace_est = kspace_under.copy()

    start = time.time()

    for _ in range(n_iter):

        # Low-rank projection
        kspace_est = lowrank_shrinkage(kspace_est, lam)

        # Data consistency
        kspace_est = mask*kspace_under + (1-mask)*kspace_est

    recon_time = time.time() - start

    # Image reconstruction
    coil_imgs = ifft2c(kspace_est)
    img = rss_combine(coil_imgs)
    img /= img.max()

    return img, img_gt, recon_time

def compute_metrics(pred, gt, lpips_model, device):

    results = {}

    # NMSE
    results["NMSE"] = np.linalg.norm(pred-gt)**2 / (np.linalg.norm(gt)**2 + 1e-8)

    # PSNR
    results["PSNR"] = compare_psnr(gt, pred, data_range=1)

    # SSIM
    results["SSIM"] = ssim(gt, pred, data_range=1)

    # HFEN
    results["HFEN"] = np.linalg.norm(
        gaussian_laplace(pred,1.5) -
        gaussian_laplace(gt,1.5)
    ) / (np.linalg.norm(gaussian_laplace(gt,1.5)) + 1e-8)

    # VIF (simplified)
    results["VIF"] = np.var(pred) / (np.var(gt) + 1e-8)

    # LPIPS
    pred_t = torch.from_numpy(pred).to(device=device, dtype=torch.float32)
    gt_t   = torch.from_numpy(gt).to(device=device, dtype=torch.float32)

    pred_t = pred_t.unsqueeze(0).unsqueeze(0)
    gt_t   = gt_t.unsqueeze(0).unsqueeze(0)

    pred_t = pred_t.repeat(1,3,1,1)*2 - 1
    gt_t   = gt_t.repeat(1,3,1,1)*2 - 1

    results["LPIPS"] = lpips_model(pred_t, gt_t).item()

    return results

def evaluate_dataset(folder, accel=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='alex').to(device)

    files = list(Path(folder).glob("*.h5"))

    results = {"NMSE":[], "PSNR":[], "SSIM":[],
               "HFEN":[], "VIF":[], "LPIPS":[], "TIME":[]}

    for file in tqdm(files):
        with h5py.File(file,"r") as hf:
            kspace_vol = hf["kspace"][:]

        for slice_k in kspace_vol:

            pred, gt, t = lowrank_pi_recon(slice_k, accel)

            metric_vals = compute_metrics(pred, gt, lpips_model, device)

            for k in metric_vals:
                results[k].append(metric_vals[k])

            results["TIME"].append(t)

    return {k:np.mean(v) for k,v in results.items()}