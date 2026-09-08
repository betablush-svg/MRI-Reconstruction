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

def spirit_self_consistency(coil_imgs):
    """
    Enforces coil consistency constraint.
    Simple spatial smoothing between coils.
    """
    avg = np.mean(coil_imgs, axis=0)
    return coil_imgs - avg

def wavelet_soft_thresh(img, lam, wavelet="db4"):
    coeffs = pywt.wavedec2(img, wavelet, level=3)
    coeffs_thresh = [coeffs[0]]

    for detail in coeffs[1:]:
        coeffs_thresh.append(tuple(
            pywt.threshold(d, lam, mode="soft") for d in detail
        ))

    return pywt.waverec2(coeffs_thresh, wavelet)

def l1_spirit_recon(full_kspace,
                    accel=4,
                    n_iter=30,
                    lam_wavelet=0.01,
                    lam_spirit=0.02,
                    step=0.6):

    kspace = to_complex_kspace(full_kspace)
    coils,H,W = kspace.shape

    # ----- Ground truth -----
    coil_imgs_full = ifft2c(kspace)
    img_gt = np.sqrt(np.sum(np.abs(coil_imgs_full)**2, axis=0))
    img_gt /= img_gt.max()

    # ----- Undersample -----
    center_fraction = 0.08 if accel==4 else 0.04
    mask = generate_vd_mask((H,W), accel, center_fraction)
    mask = mask[np.newaxis,:,:]

    kspace_under = kspace * mask

    # ----- Initial guess -----
    coil_imgs = ifft2c(kspace_under)

    start = time.time()

    for _ in range(n_iter):

        # Forward model
        k_est = fft2c(coil_imgs)

        # Data consistency
        k_est = mask*kspace_under + (1-mask)*k_est
        coil_imgs = ifft2c(k_est)

        # SPIRiT self-consistency
        coil_imgs = coil_imgs - lam_spirit*spirit_self_consistency(coil_imgs)

        # ℓ1 Wavelet sparsity (image domain)
        img = np.sqrt(np.sum(np.abs(coil_imgs)**2, axis=0))
        img = wavelet_soft_thresh(img, lam_wavelet)

        # Back-project image to coils
        coil_imgs = np.repeat(img[np.newaxis,:,:], coils, axis=0)

    recon_time = time.time()-start

    img = np.sqrt(np.sum(np.abs(coil_imgs)**2, axis=0))
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
            pred, gt, t = l1_spirit_recon(slice_k, accel)

            results["NMSE"].append(nmse(pred,gt))
            results["PSNR"].append(compare_psnr(gt,pred,data_range=1))
            results["SSIM"].append(ssim(gt,pred,data_range=1))
            results["HFEN"].append(hfen(pred,gt))
            results["VIF"].append(vif(pred,gt))
            results["LPIPS"].append(compute_lpips(pred,gt))
            results["TIME"].append(t)

    return {k:np.mean(v) for k,v in results.items()}