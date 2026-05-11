import argparse
import os
import sys
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from typing import *
from matplotlib import pyplot as plt
from pathlib import Path
from torch.cuda.amp import autocast as autocast
import h5py
from fiber_tracking_and_segment.segmentation.GeoGridNet.utils.visualize_fibers import plot_fibers_with_labels
from utils.metrics_plots import save_confusion_matrix, calculate_metrics
from ASGM import adaptive_spatial_grid_mapping
from model_dual_branch import FusionNet

expected_tracts = [
    'UF_right', 'UF_left',    # Uncinate Fasciculus
    'AF_right', 'AF_left',    # Arcuate Fasciculus
    'IFO_right', 'IFO_left',  # Inferior Fronto-Occipital
    'SLF_III_right', 'SLF_III_left'   # Superior Longitudinal Fasciculus
]

def predict_in_batches(model, points, fibermaps, labels, batch_size=32, device='cuda:0'):
    """
    Perform predictions in batches to avoid memory overflow.
    """
    model.eval()
    N = points.shape[0]
    all_preds = []
    
    # If the number of fibres is less than the batch size, process directly.
    if N <= batch_size:
        with torch.no_grad():
            points_batch = points.to(device)
            fibermaps_batch = fibermaps.to(device)
            pred = model(points_batch, fibermaps_batch) 
            return pred
    else:
        num_batches = (N + batch_size - 1) // batch_size
        batch_preds = []
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"Batch predict (batch_size={batch_size})", leave=False):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, N)
                
                # Retrieve the current batch
                points_batch = points[start_idx:end_idx].to(device)
                fibermaps_batch = fibermaps[start_idx:end_idx].to(device)

                # Reduce memory usage by employing automatic precision mixing
                with autocast():
                    preds_batch = model(points_batch, fibermaps_batch) 
                
                batch_preds.append(preds_batch.cpu())
                
                # Clear the GPU cache
                del points_batch, fibermaps_batch
                if i % 5 == 0:  
                    torch.cuda.empty_cache()
        
        all_preds = torch.cat(batch_preds, dim=0)
        torch.cuda.empty_cache()
        
        return all_preds.to(device)
            
def iter_one_sample(model: torch.nn.Module,
                   device,
                   root):

    model.eval()
    sample_list = sorted(os.listdir(root))
    mac_precision_list, mac_recall_list, mac_f1_list, accuracy_list = [], [], [], []

    for sample_idx, sample_name in enumerate(tqdm(sample_list, desc="processing")):
        sample_path = os.path.join(root, sample_name)
        if not os.path.isdir(sample_path):
            continue

        features_path = os.path.join(sample_path, "8_features.h5")
        labels_path = os.path.join(sample_path, "8_labels.h5")
        
        if not (os.path.exists(features_path) and os.path.exists(labels_path)):
            continue
        
        with h5py.File(features_path, 'r') as f:
            features = f['features'][:]  # [N_i, 15, 3]
            
        with h5py.File(labels_path, 'r') as f:
            labels = f['labels'][:]      # [N_i,]
            
        if features.shape[0] != labels.shape[0]:
            continue
        nan_count = np.isnan(features).sum()
        
        if nan_count:
            print(f"Warning: The sample {sample_name} contains NaN values.")
            continue 

        features = np.array(features)  # [N, 15, 3]
        labels = np.array(labels)      # [N,]
        grid_size=(32, 32)
        fibermaps = adaptive_spatial_grid_mapping(
                        features,
                        grid_size=grid_size,
                        use_interpolation=False,
                        use_original_coords=True,
                        normalize_by_fiber=True
                    )  # [N_i, H, W, 3]
        
        fibermaps = torch.from_numpy(fibermaps.astype(np.float32))  # [N, 15, 3]
        fibermaps = fibermaps.permute(0,3,1,2)
        labels = torch.from_numpy(labels.astype(np.int64))   # [N,]
        point_sets = torch.from_numpy(features.astype(np.float32))

        if labels.dim() == 2:
            labels = labels[:, 0]  # [B,1] -> [B]
            
        fibermaps, point_sets, labels = fibermaps.to(device), point_sets.to(device), labels.to(device)

        with torch.no_grad():
            pred = predict_in_batches(model, point_sets, fibermaps, labels)  
                
        labels = labels.cpu().detach().numpy() - 1
        
        _, pred_idx = torch.max(pred, dim=1)
        pred_idx = pred_idx.cpu().detach().numpy()
        
        confusion_matrix_path = os.path.join(sample_path, "confusion_matrix.png")

        mac_precision, mac_recall, mac_f1, accuracy = calculate_metrics(labels.tolist(), pred_idx.tolist())
        save_confusion_matrix(labels, pred_idx, confusion_matrix_path, class_names = expected_tracts)
        
        mac_precision_list.append(mac_precision)
        mac_recall_list.append(mac_recall)
        mac_f1_list.append(mac_f1)
        accuracy_list.append(accuracy)
        
        pred_idx = pred_idx + 1
        h5_preds_path = os.path.join(sample_path,  "8_preds_dualbranch.h5")
        
        with h5py.File(h5_preds_path, 'w') as f:
            f.create_dataset('labels', data=pred_idx, compression='gzip')
            print(f"  predict results are saved to: {h5_preds_path}")
        
        # output_path_label =os.path.join(sample_path, "fiber_visualization_gt.png")
        output_path_pred =os.path.join(sample_path, "fiber_visualization_dualbranch.png")

        # plot_fibers_with_labels(features_path, labels_path, expected_tracts, output_path=output_path_label)
        plot_fibers_with_labels(features_path, h5_preds_path, expected_tracts, output_path=output_path_pred)

    nums = len(mac_precision_list)
    print(f"finish! mac_precision:{sum(mac_precision_list)/nums},mac_recall:{sum(mac_recall_list)/nums},mac_f1:{sum(mac_f1_list)/nums},accuracy:{sum(accuracy_list)/nums}")
    
if __name__ == '__main__':
    device = "cuda:0"
    checkpoint_path = ""
    num_classes = 8
    root = ""
    
    checkpoints = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=True)
    
    model = FusionNet(num_classes=8,
                 use_arc_encoding=True,
                 use_attention=True,
                 spatial_base_channels=32,
                 spatial_num_blocks=4,
                 fusion_method='cross_attention', 
                 fusion_dim=512,
                 dropout_rate=0.3)
    
    model.to(device)
    if checkpoints['model_state_dict'] is not None:
        model.load_state_dict(checkpoints['model_state_dict'])
    
    iter_one_sample(model,
                   device,
                   root)