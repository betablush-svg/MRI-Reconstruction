import torch
from utils import dataloader
import time
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import shutil

from models import unet, kiki, admm
from models import deep_cascade as dc
from models import dual_domain as dd
from models import varnet, modl
from models import refinegan as rg
from models import transformer as trf
from models import grappa, sense
import matplotlib.pyplot as plt

def to_magnitude(x):
    if x.shape[1] == 2:  # real + imag
        return torch.sqrt(x[:,0]**2 + x[:,1]**2)
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file = "file1000021.h5"
raw_file_path = "datasets\\knee_multicoil_train_batch_0\\multicoil_train\\" + file
file_path = "datasets\\knee_multicoil_undersampled_x4\\"+file
sample_path = "datasets\\sample"

# Construct the full destination path
filename = os.path.basename(file_path)
dest_path = os.path.join(sample_path, filename)

# Move only if the file does not already exist
if not os.path.exists(dest_path):
    shutil.copy(file_path, dest_path)
    print(f"Moved: {filename}")
else:
    print(f"Skipped: {filename} (Already exists)")

test_set = dataloader.MRIDataset(sample_path)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False)
print("No. of Slices = ",len(test_loader))

model1 = unet.PhysicsUNet().to(device)
model4 = kiki.KIKINet().to(device)
model5 = dc.DeepCascadeCNN().to(device)
model6 = dd.DualDomainNet(stages=5).to(device)
model7 = admm.ADMMNet().to(device)
model8 = varnet.VarNet(cascades=8).to(device)
model9 = modl.MoDL(iterations=8).to(device)
model10 = rg.RefineGAN().to(device)
model11 = trf.TransformerReconstruction(stages=4).to(device)

model1.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\UNET_best_model_x4.pt", map_location=device))
model4.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\KIKI_best_model_x4.pt", map_location=device))
model5.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\DeepCascade_x4.pt", map_location=device))
model6.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\DualDomain_best_model_x4.pt", map_location=device))
model7.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\ADMM_x4.pt", map_location=device))
model8.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\VARNET_best_model_x4.pt", map_location=device))
model9.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\MODL_best_model_x4.pt", map_location=device))
model10.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\RefineGAN_best_generator_x4.pt", map_location=device))
model11.load_state_dict(torch.load("checkpoints\\ALL_MODELS\\Transformer_best_model_x4.pt", map_location=device))

models = {
    "GT": None,
    "UNet": model1,
    "KIKI": model4,
    "Deep_Cascade": model5,
    "Dual_Domain": model6,
    "ADMM": model7,
    "VARNET": model8,
    "MoDL": model9,
    "RefineGAN": model10,
    "Transformer": model11
}

save_root = "Images\\file1000021\\perturbations\\gaussian_noise_results"

'''image_level = ['Gaussian_noise','Motion_blur','Bias_field']
k_space_level = ['Complex_Gaussian_noise','Spike_noise','Random_line_dropouts']
mask_level = ['Cartesian','Radial','Poisson_disc']

Perturbation	Levels
Gaussian noise	1%, 3%, 5%, 10%
Motion blur	mild/moderate/severe
Line dropout	5%, 10%, 20%
Bias field	weak/strong
Acceleration factor	4x, 8x, 12x'''


indices = [0, 8, 16, 24, 32]

# Noise levels
noise_levels = {
    "1_percent": 0.01,
    "3_percent": 0.03,
    "5_percent": 0.05,
    "10_percent": 0.10
}

for noise_name, noise_std in noise_levels.items():

    print(f"\nApplying Gaussian Noise: {noise_name}")

    for name, model in models.items():

        all_recons = []

        # Create directory:
        # gaussian_noise_results/1_percent/UNet/
        model_dir = os.path.join(
            save_root,
            noise_name,
            name
        )

        os.makedirs(model_dir, exist_ok=True)

        with torch.no_grad():

            for sample in test_loader:

                sample = {
                    k: v.to(device)
                    for k, v in sample.items()
                }

                input_data = sample['img_zf']
                target = sample['target']
                mask = sample['mask']
                k_under = sample['kspace_under']
                k_space = sample['kspace_2ch']

                # -----------------------------------
                # Apply Gaussian Noise Perturbation
                # -----------------------------------

                noisy_input = input_data + (
                    torch.randn_like(input_data) * noise_std
                )

                noisy_kspace = k_space + (
                    torch.randn_like(k_space) * noise_std
                )

                # -----------------------------------
                # Reconstruction
                # -----------------------------------

                if name == 'GT':

                    recon = target.squeeze().cpu().numpy()

                elif name == "KIKI":

                    out = model(noisy_kspace)

                    recon = (
                        to_magnitude(out)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                else:

                    out = model(
                        noisy_input,
                        k_under,
                        mask
                    )

                    recon = (
                        to_magnitude(out)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                all_recons.append(recon)

        # Stack slices
        imgs = np.stack(all_recons, axis=0)

        # -----------------------------------
        # Save Selected Slices
        # -----------------------------------

        for idx in indices:

            slice_img = imgs[idx]

            # Normalize for visualization
            slice_img = (
                slice_img - slice_img.min()
            ) / (
                slice_img.max() - slice_img.min() + 1e-8
            )

            save_path = os.path.join(
                model_dir,
                f"slice_{idx:03d}.png"
            )

            plt.imsave(
                save_path,
                slice_img,
                cmap='gray'
            )

        print(f"Saved {name} - {noise_name}")