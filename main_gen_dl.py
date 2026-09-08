import torch
from utils.saving import save_results, print_metrics
from utils import dataloader

from models import transformer, swin, gan
from models import dual_domain as dd
from models import self_supervised as ssd
from models import diffusion
from models import da_gan as dagan
from models import refinegan as rgan

import gc


## Undersampled Data x4 and x8
undersampled_train_x8_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x8/train"
undersampled_val_x8_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x8/val"
undersampled_test_x8_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x8/test"

undersampled_train_x4_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x4/train"
undersampled_val_x4_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x4/val"
undersampled_test_x4_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x4/test"

## Multicoil RAW
raw_multicoil_train_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_raw/train"
raw_multicoil_val_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_raw/val"
raw_multicoil_test_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_raw/test"

train_set_x4 = dataloader.MRIDataset(undersampled_train_x4_path)
val_set_x4 = dataloader.MRIDataset(undersampled_val_x4_path)
test_set_x4 = dataloader.MRIDataset(undersampled_test_x4_path)

train_loader_x4 = torch.utils.data.DataLoader(train_set_x4,batch_size=4,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=2)
val_loader_x4 = torch.utils.data.DataLoader(val_set_x4,batch_size=4,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=False,prefetch_factor=2)
test_loader_x4   = torch.utils.data.DataLoader(test_set_x4,batch_size=1,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=False,prefetch_factor=2)

train_set_x8 = dataloader.MRIDataset(undersampled_train_x8_path)
val_set_x8 = dataloader.MRIDataset(undersampled_val_x8_path)
test_set_x8 = dataloader.MRIDataset(undersampled_test_x8_path)

train_loader_x8 = torch.utils.data.DataLoader(train_set_x8,batch_size=4,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=2)
val_loader_x8 = torch.utils.data.DataLoader(val_set_x8,batch_size=4,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=False,prefetch_factor=2)
test_loader_x8   = torch.utils.data.DataLoader(test_set_x8,batch_size=1,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=False,prefetch_factor=2)


# epochs
epochs = 50

def run_recon_experiment(method_name):

    print(f"\n===== Running {method_name} =====")

    if method_name == "DualDomain":
        model = dd.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = dd.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()
        model = dd.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = dd.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()
        

    elif method_name == "Transformer":
        model = transformer.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = transformer.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()'''
        model = transformer.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = transformer.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()
        
    elif method_name == "Swin":
        model = swin.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = swin.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = swin.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = swin.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "SelfSupervised":
        model = ssd.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = ssd.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = ssd.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = ssd.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "GAN":
        model = gan.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = gan.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = gan.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = gan.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "Diffusion":
        model, alpha, alpha_bar = diffusion.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = diffusion.evaluate_model(model,
                                     test_loader_x4,
                                     alpha,
                                     alpha_bar)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        del alpha
        del alpha_bar
        torch.cuda.empty_cache()
        gc.collect() 
        model, alpha, alpha_bar = diffusion.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = diffusion.evaluate_model(model,
                                     test_loader_x8,
                                     alpha,
                                     alpha_bar)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        del alpha
        del alpha_bar
        torch.cuda.empty_cache()
        gc.collect()         
    
    elif method_name == "DAGAN":
        model = dagan.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = dagan.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()         
        model = dagan.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = dagan.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()         

    elif method_name == "RefineGAN":
        model = rgan.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = rgan.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()       
        model = rgan.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = rgan.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        
    else:
        raise ValueError("Unknown method")


recon_methods = [
    "Diffusion",
    "DualDomain",
    "Transformer",
    "Swin",
    "SelfSupervised",
    "GAN",
    "DAGAN",
    "RefineGAN"
]

for method in recon_methods:
    run_recon_experiment(method)