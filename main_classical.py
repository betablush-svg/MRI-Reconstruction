import torch
from utils.saving import save_results, print_metrics
from models import ZFRecon
from models import grappa
from models import sense
from models import low_rank_dictionary as LR
from models import cs_wavelet as csw
from models import cs_tv as cstv
from models import sparse_sense as ss
from models import l1_spirit as l1
from models import pocs
import gc

## Undersampled Data x4 and x8
undersampled_test_x8_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x8/test"

undersampled_test_x4_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_undersampled_x4/test"

## Multicoil RAW
raw_multicoil_test_path = "/datasets/tir-dataset-mri-recon/knee_multicoil_raw/test"

def run_recon_experiment(method_name):

    print(f"\n===== Running {method_name} =====")

    if method_name=="ZFRecon":
        metrics = ZFRecon.evaluate_image_dataset(undersampled_test_x4_path)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
        metrics = ZFRecon.evaluate_image_dataset(undersampled_test_x8_path)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()        
    
    elif method_name=="GRAPPA":
        metrics = grappa.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()  

        metrics = grappa.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          

    elif method_name=="SENSE":
        metrics = sense.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          
        metrics = sense.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          

    elif method_name=="LowRankDictionary":
        metrics = LR.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          
        metrics = LR.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()  

    elif method_name=="CSWavelength":
        metrics = csw.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          
        metrics = csw.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()  

    elif method_name=="CS_TV":
        metrics = cstv.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          
        metrics = cstv.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          

    elif method_name=="SPARSE_SENSE":
        metrics = ss.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          
        metrics = ss.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          

    elif method_name=="L1_SPIRIT":
        metrics = l1.evaluate_dataset(raw_multicoil_test_path, accel=4)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          
        metrics = l1.evaluate_dataset(raw_multicoil_test_path, accel=8)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          

    elif method_name=="POCS":
        metrics = pocs.evaluate_dataset(undersampled_test_x4_path)
        print_metrics(method_name+"_x4", metrics)
        save_results(method_name+"_x4", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          
        metrics = pocs.evaluate_dataset(undersampled_test_x8_path)
        print_metrics(method_name+"_x8", metrics)
        save_results(method_name+"_x8", metrics)
        del metrics
        torch.cuda.empty_cache()
        gc.collect()          

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
   "POCS"
]

for method in recon_methods:
    run_recon_experiment(method)