import json
import os
import numpy as np
import torch

def convert_to_serializable(obj):

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item()

    return obj


def save_results(method_name, metrics, model=None):

    os.makedirs("datasets\\new_results\\metrics", exist_ok=True)
    os.makedirs("datasets\\new_results\\checkpoints", exist_ok=True)

    # Convert dictionary safely
    metrics_clean = {
        k: convert_to_serializable(v)
        for k, v in metrics.items()
    }

    with open(f"datasets\\new_results\\metrics\\{method_name}.json", "w") as f:
        json.dump(metrics_clean, f, indent=4)

    if model is not None:
        torch.save(model.state_dict(),
                   f"datasets\\new_results\\checkpoints\\{method_name}.pt")
        
def print_metrics(method_name, metrics):
    print(f"\n===== Metrics {method_name} =====")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")