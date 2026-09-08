import torch
from utils.saving import save_results, print_metrics
from utils import dataloader

from models import dncnn, srcnn, unet
from models import deep_cascade , automap, kiki

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

    if method_name == "DnCNN":
        model = dncnn.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = dncnn.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = dncnn.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = dncnn.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "SRCNN":
        model = srcnn.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = srcnn.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = srcnn.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = srcnn.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "DeepCascade":
        model = deep_cascade.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = deep_cascade.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = deep_cascade.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = deep_cascade.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "UNet":
        model = unet.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = unet.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = unet.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = unet.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "AUTOMAP":
        model = automap.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = automap.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        model = automap.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = automap.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        

    elif method_name == "KIKI":
        '''
        model = kiki.train_model(train_loader_x4, val_loader_x4, epochs)
        metrics = kiki.evaluate_model(model, test_loader_x4)
        save_results(method_name+"_x4", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()
        '''        
        model = kiki.train_model(train_loader_x8, val_loader_x8, epochs)
        metrics = kiki.evaluate_model(model, test_loader_x8)
        save_results(method_name+"_x8", metrics, model)
        del model
        del metrics
        torch.cuda.empty_cache()
        gc.collect()   
             
        
    else:
        raise ValueError("Unknown method")


recon_methods = [
    "DnCNN"
#    "SRCNN",
#    "DeepCascade",
#    "UNet",
#    "AUTOMAP",
#    "KIKI"
]

for method in recon_methods:
    run_recon_experiment(method)