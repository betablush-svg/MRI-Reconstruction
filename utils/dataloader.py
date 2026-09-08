import h5py
import torch
import numpy as np
from pathlib import Path


def center_crop(img, size=320):

    H, W = img.shape

    top = (H - size) // 2
    left = (W - size) // 2

    return img[top:top+size, left:left+size]


def create_mask(shape, accel=4):

    H, W = shape

    mask = np.zeros((H,W))

    center_fraction = 0.08
    num_low = int(W * center_fraction)

    center = W//2

    mask[:,center-num_low//2:center+num_low//2] = 1

    prob = (W/accel - num_low) / (W - num_low)

    random_lines = np.random.rand(W) < prob

    mask[:,random_lines] = 1

    return mask


class MRIDataset(torch.utils.data.Dataset):

    def __init__(self, folder, crop_size=320, accel=4):

        self.files = list(Path(folder).glob("*.h5"))

        self.examples = []

        self.crop_size = crop_size
        self.accel = accel

        for f in self.files:

            with h5py.File(f,"r") as hf:

                n = hf["recon_target"].shape[0]

            for i in range(n):

                self.examples.append((f,i))


    def __len__(self):

        return len(self.examples)


    def __getitem__(self, idx):

        f,i = self.examples[idx]

        with h5py.File(f,"r") as hf:

            target = hf["recon_target"][i]

        # --------------------------
        # Normalize
        # --------------------------

        target = target / (target.max() + 1e-8)

        target = center_crop(target, self.crop_size)

        target = torch.tensor(target).float()

        # --------------------------
        # Full k-space
        # --------------------------

        kspace_full = torch.fft.fft2(target, norm="ortho")

        # --------------------------
        # Undersampling mask
        # --------------------------

        mask = create_mask(target.shape, self.accel)

        mask = torch.tensor(mask).float()

        # --------------------------
        # Undersampled k-space
        # --------------------------

        kspace_under = kspace_full * mask

        # --------------------------
        # Zero-filled reconstruction
        # --------------------------

        img_zf = torch.fft.ifft2(kspace_under, norm="ortho").real

        img_zf = img_zf / (torch.max(torch.abs(img_zf)) + 1e-8)

        # --------------------------
        # Real + Imag channels
        # --------------------------

        k_real = torch.real(kspace_under)
        k_imag = torch.imag(kspace_under)

        kspace_2ch = torch.stack([k_real, k_imag], dim=0)

        # --------------------------
        # Add channel dims
        # --------------------------

        img_zf = img_zf.unsqueeze(0)
        target = target.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return {

            "img_zf": img_zf,           # (1,H,W)

            "target": target,           # (1,H,W)

            "kspace_under": kspace_under, # complex

            "kspace_2ch": kspace_2ch,   # (2,H,W)

            "mask": mask                # (1,H,W)

        }