import torch
from utils.saving import save_results
from models import ZFRecon
from models import grappa
from models import sense
from models import low_rank_dictionary as LR
from models import cs_wavelet as csw
from models import cs_tv as cstv
from models import sparse_sense as ss
from models import l1_spirit as l1

from models import dncnn, srcnn, unet
from models import ista_net as ista
from models import varnet, modl, transformer, swin, gan
from models import dual_domain as dd
from models import self_supervised as ssd
from models import diffusion
from models import plug_and_play as pp

## Undersampled Data x4 and x8
undersampled_train_x8_path = "datasets\\knee_multicoil_small_undersampled_x8\\train"
undersampled_val_x8_path = "datasets\\knee_multicoil_small_undersampled_x8\\val"
undersampled_test_x8_path = "datasets\\knee_multicoil_small_undersampled_x8\\test"

undersampled_train_x4_path = "datasets\\knee_multicoil_small_undersampled_x4\\train"
undersampled_val_x4_path = "datasets\\knee_multicoil_small_undersampled_x4\\val"
undersampled_test_x4_path = "datasets\\knee_multicoil_small_undersampled_x4\\test"

## Multicoil RAW
raw_multicoil_train_path = "datasets\\knee_multicoil_small\\train"
raw_multicoil_val_path = "datasets\\knee_multicoil_small\\val"
raw_multicoil_test_path = "datasets\\knee_multicoil_small\\test"

train_loader_x4 = torch.utils.data.DataLoader(undersampled_train_x4_path,batch_size=8,shuffle=True)
val_loader_x4 = torch.utils.data.DataLoader(undersampled_val_x4_path,batch_size=8,shuffle=True)
test_loader_x4   = torch.utils.data.DataLoader(undersampled_test_x4_path,batch_size=1)

train_loader_x8 = torch.utils.data.DataLoader(undersampled_train_x8_path,batch_size=8,shuffle=True)
val_loader_x8 = torch.utils.data.DataLoader(undersampled_val_x8_path,batch_size=8,shuffle=True)
test_loader_x8   = torch.utils.data.DataLoader(undersampled_test_x8_path,batch_size=1)

# epochs
epochs = 1

def print_metrics(method_name, metrics):
    print(f"\n===== Metrics {method_name} =====")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

def run_recon_experiment(method_name):

    print(f"\n===== Running {method_name} =====")

    if method_name=="ZFRecon":
        metrics = ZFRecon.evaluate_image_dataset(undersampled_test_x4_path)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = ZFRecon.evaluate_image_dataset(undersampled_test_x8_path)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
    
    elif method_name=="GRAPPA":
        metrics = grappa.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = grappa.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)

    elif method_name=="SENSE":
        metrics = sense.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = sense.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)

    elif method_name=="LowRankDictionary":
        metrics = LR.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = LR.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)

    elif method_name=="CSWavelength":
        metrics = csw.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = csw.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)

    elif method_name=="CS_TV":
        metrics = cstv.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = cstv.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)

    elif method_name=="SPARSE_SENSE":
        metrics = ss.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = ss.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)

    elif method_name=="L1_SPIRIT":
        metrics = l1.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        metrics = l1.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)

    elif method_name == "DnCNN":
        model = dncnn.train_model(train_loader_x4, epochs)
        metrics = dncnn.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = dncnn.train_model(train_loader_x8, epochs)
        metrics = dncnn.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "SRCNN":
        model = srcnn.train_model(train_loader_x4, epochs)
        metrics = srcnn.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = srcnn.train_model(train_loader_x8, epochs)
        metrics = srcnn.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "UNet":
        model = unet.train_model(train_loader_x4, epochs)
        metrics = unet.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = unet.train_model(train_loader_x8, epochs)
        metrics = unet.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "ISTA-NET":
        model = ista.train_model(train_loader_x4, epochs)
        metrics = ista.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = ista.train_model(train_loader_x8, epochs)
        metrics = ista.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "VARNET":
        model = varnet.train_model(train_loader_x4, epochs)
        metrics = varnet.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = varnet.train_model(train_loader_x8, epochs)
        metrics = varnet.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "MoDL":
        model = modl.train_model(train_loader_x4, epochs)
        metrics = modl.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = modl.train_model(train_loader_x8, epochs)
        metrics = modl.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "DualDomain":
        model = dd.train_model(train_loader_x4, epochs)
        metrics = dd.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = dd.train_model(train_loader_x8, epochs)
        metrics = dd.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "Transformer":
        model = transformer.train_model(train_loader_x4, epochs)
        metrics = transformer.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = transformer.train_model(train_loader_x8, epochs)
        metrics = transformer.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "Swin":
        model = swin.train_model(train_loader_x4, epochs)
        metrics = swin.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = swin.train_model(train_loader_x8, epochs)
        metrics = swin.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "SelfSupervised":
        model = ssd.train_model(train_loader_x4, epochs)
        metrics = ssd.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = ssd.train_model(train_loader_x8, epochs)
        metrics = ssd.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "GAN":
        model = gan.train_model(train_loader_x4, epochs)
        metrics = gan.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = gan.train_model(train_loader_x8, epochs)
        metrics = gan.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "Diffusion":
        model, alpha, alpha_bar = diffusion.train_model(train_loader_x4, epochs)
        metrics = diffusion.evaluate_model(model,
                                     test_loader_x4,
                                     alpha,
                                     alpha_bar)
        save_results(method_name+"_x4", metrics, model)
        model, alpha, alpha_bar = diffusion.train_model(train_loader_x8, epochs)
        metrics = diffusion.evaluate_model(model,
                                     test_loader_x8,
                                     alpha,
                                     alpha_bar)
        save_results(method_name+"_x8", metrics, model)

    elif method_name == "PnP":
        model = pp.train_model(train_loader_x4, epochs)
        metrics = pp.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        model = pp.train_model(train_loader_x8, epochs)
        metrics = pp.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        
    else:
        raise ValueError("Unknown method")


recon_methods = [
    "ZFRecon",
    "GRAPPA",
    "SENSE",
    "LowRankDictionary",
    "CSWavelength",
    "CS_TV",
    "SPARSE_SENSE",
    "L1_SPIRIT",
    "DnCNN",
    "SRCNN",
    "UNet",
    "ISTA-NET",
    "VARNET",
    "MoDL",
    "DualDomain",
    "Transformer",
    "Swin",
    "SelfSupervised",
    "GAN",
    "Diffusion",
    "PnP"
]

for method in recon_methods:
    run_recon_experiment(method)