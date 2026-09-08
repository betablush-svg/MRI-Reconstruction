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

def data_consistency(x, k_under, mask):

    x_k = torch.fft.fft2(x, norm='ortho')
    x_k = mask * k_under + (1-mask) * x_k
    x = torch.fft.ifft2(x_k, norm='ortho').real

    return x

def image_projection(x):

    # positivity constraint
    x = torch.clamp(x, min=0)
    return x


def pocs_reconstruction(k_under, mask, iterations=20):

    device = k_under.device

    # Zero-filled initialization
    x = torch.fft.ifft2(k_under, norm='ortho').real

    for _ in range(iterations):

        # Projection onto image constraint
        x = image_projection(x)

        # Projection onto data consistency
        x = data_consistency(x, k_under, mask)

    return x

class MRITestDataset(torch.utils.data.Dataset):

    def __init__(self, folder, acceleration=4):

        self.files = list(Path(folder).glob("*.h5"))
        self.examples = []
        self.acceleration = acceleration

        for f in self.files:
            with h5py.File(f, "r") as hf:
                n = hf["recon_target"].shape[0]

            for i in range(n):
                self.examples.append((f, i))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        f, i = self.examples[idx]

        with h5py.File(f, "r") as hf:
            tgt = hf["recon_target"][i]

        tgt = tgt / tgt.max()

        tgt = torch.tensor(tgt).float()

        # FFT to k-space
        kspace = torch.fft.fft2(tgt, norm="ortho")

        # Cartesian undersampling mask
        mask = torch.zeros_like(kspace)
        mask[:, ::self.acceleration] = 1

        k_under = mask * kspace

        return (
            tgt.unsqueeze(0),
            k_under,
            mask
        )
    
def evaluate_pocs(loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_model = lpips.LPIPS(net='alex').to(device)

    results = {
        "NMSE":[],
        "PSNR":[],
        "SSIM":[],
        "HFEN":[],
        "VIF":[],
        "LPIPS":[],
        "TIME":[]
    }

    for tgt, k_under, mask in loader:

        k_under = k_under.to(device)
        mask = mask.to(device)

        start = time.time()

        recon = pocs_reconstruction(k_under[0], mask[0])

        recon_time = time.time() - start

        pred = recon.detach().cpu().numpy()
        gt = tgt[0,0].numpy()

        pred = pred/(pred.max()+1e-8)
        gt = gt/(gt.max()+1e-8)

        nmse_val = np.linalg.norm(pred-gt)**2 / (np.linalg.norm(gt)**2+1e-8)

        results["NMSE"].append(nmse_val)
        results["PSNR"].append(compare_psnr(gt,pred,data_range=1))
        results["SSIM"].append(ssim(gt,pred,data_range=1,win_size=7))

        results["HFEN"].append(
            np.linalg.norm(
                gaussian_laplace(pred,1.5)
                - gaussian_laplace(gt,1.5)
            ) /
            (np.linalg.norm(gaussian_laplace(gt,1.5))+1e-8)
        )

        results["VIF"].append(
            np.var(pred)/(np.var(gt)+1e-8)
        )

        pred_lp = torch.tensor(pred[None,None,:,:]).repeat(1,3,1,1).float().to(device)
        gt_lp = torch.tensor(gt[None,None,:,:]).repeat(1,3,1,1).float().to(device)

        pred_lp = pred_lp*2 - 1
        gt_lp = gt_lp*2 - 1

        lpips_val = lpips_model(pred_lp, gt_lp).mean().item()

        results["LPIPS"].append(lpips_val)
        results["TIME"].append(recon_time)

    return {k: float(np.mean(v)) for k,v in results.items()}

def evaluate_dataset(path):
    test_set = MRITestDataset(
        path,
        acceleration=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1
    )
    metrics = evaluate_pocs(test_loader)
    return metrics