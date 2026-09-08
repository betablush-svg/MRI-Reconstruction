import h5py
import numpy as np
import torch
import pywt
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

def wavelet_soft_thresh(img, lam, wavelet="db4"):
    coeffs = pywt.wavedec2(img, wavelet, level=3)
    coeffs_thresh = [coeffs[0]]

    for detail in coeffs[1:]:
        coeffs_thresh.append(tuple(
            pywt.threshold(d, lam, mode="soft") for d in detail
        ))

    return pywt.waverec2(coeffs_thresh, wavelet)

def tv_denoise(img, weight=0.01, n_iter=10):
    for _ in range(n_iter):
        dx = np.roll(img,-1,axis=1) - img
        dy = np.roll(img,-1,axis=0) - img
        img += weight*(dx+dy)
    return img

def cs_wavelet_recon(full_kspace, accel=4, n_iter=20, lam=0.01):

    kspace = to_complex_kspace(full_kspace)
    coils,H,W = kspace.shape

    # ----- Ground truth -----
    coil_imgs = ifft2c(kspace)
    img_gt = rss_combine(coil_imgs)
    img_gt /= img_gt.max()

    # ----- Undersampling -----
    center_fraction = 0.08 if accel==4 else 0.04
    mask = generate_vd_mask((H,W), accel, center_fraction)
    kspace_under = kspace * mask

    # ----- Initial guess (ZF) -----
    img = rss_combine(ifft2c(kspace_under))

    start = time.time()

    for _ in range(n_iter):

        # Data consistency
        k_est = fft2c(img)
        k_est = mask*kspace_under[0] + (1-mask)*k_est

        img = np.abs(ifft2c(k_est))

        # Wavelet sparsity
        img = wavelet_soft_thresh(img, lam)

        # TV regularization
        img = tv_denoise(img)

    recon_time = time.time()-start
    img /= img.max()

    return img, img_gt, recon_time

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
            pred, gt, t = cs_wavelet_recon(slice_k, accel)

            results["NMSE"].append(nmse(pred,gt))
            results["PSNR"].append(compare_psnr(gt,pred,data_range=1))
            results["SSIM"].append(ssim(gt,pred,data_range=1))
            results["HFEN"].append(hfen(pred,gt))
            results["VIF"].append(vif(pred,gt))
            results["LPIPS"].append(compute_lpips(pred,gt))
            results["TIME"].append(t)

    return results
