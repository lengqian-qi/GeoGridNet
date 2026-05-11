import os
import json
import numpy as np
import nibabel as nib
import h5py
from scipy.interpolate import interp1d
from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_features_from_trk(trk_file, num_points=15):
    """
    Extract RAS features directly from TRK files
    Parameters:
        trk_file: Path to TRK file
        num_points: Number of sampling points per fiber
    Returns:
        feat: Feature array [N, num_points, 3]
    """

    trk = nib.streamlines.load(trk_file)
    streamlines = trk.streamlines  
    num_fibers = len(streamlines)
    
    feat = np.zeros((num_fibers, num_points, 3))  # [N, num_points, 3]
    
    #Resampling each fiber
    for i, fiber in enumerate(streamlines):
        if i % 10000 == 0:
            print(f"Processing fibers {i}/{num_fibers}")
        
        n_points = len(fiber)
        if n_points <= 1:
            continue  
            
        # Create an Interpolator (Linear Interpolation)
        t_original = np.linspace(0, 1, n_points)
        t_target = np.linspace(0, 1, num_points)
        
        for dim in range(3):
            interpolator = interp1d(t_original, fiber[:, dim], 
                                   kind='linear', 
                                   bounds_error=False,
                                   fill_value="extrapolate")
            feat[i, :, dim] = interpolator(t_target)
    
    return feat, streamlines

def process_sample(root_path, expected_tracts, label_mapping, out_sample_dir, num_points=15):
    """
    Processing individual sample folders
    Parameters:
        root_path: Path to the sample folder
        expected_tracts: List of expected fibre bundle names
        label_mapping: Label mapping dictionary
        num_points: Number of sampling points per fibre
    Returns:
        all_features: Merged feature array [8*N, num_points, 3]
        all_labels: Merged label array [8*N,]
    """
    print(f"\processing: {os.path.basename(root_path)}")
    print("="*60)
    os.makedirs(out_sample_dir, exist_ok=True)

    all_features_list = []
    all_labels_list = []
    
    for trk_name in expected_tracts:
        trk_path = os.path.join(root_path, trk_name + ".trk")
                
        # extraction
        features, _ = extract_features_from_trk(trk_path, num_points)
        num_fibers = features.shape[0]
        
        # Retrieve the label from the global mapping
        label = label_mapping[trk_name]
        labels = np.full(num_fibers, label, dtype=np.int32)
        
        all_features_list.append(features)
        all_labels_list.append(labels)
        
        print(f" {trk_name}: {num_fibers}fibers, label={label}")
    

    # Merge all features and tags
    all_features = np.vstack(all_features_list)  # [8*N, num_points, 3]
    all_labels = np.concatenate(all_labels_list)  # [8*N,]
    
    print(f"features.shape: {all_features.shape}")
    print(f"labels.shape: {all_labels.shape}")
    
    h5_features_path = os.path.join(out_sample_dir,  "8_features.h5")
    h5_labels_path = os.path.join(out_sample_dir, "8_labels.h5")
    
    with h5py.File(h5_features_path, 'w') as f:
        f.create_dataset('features', data=all_features, compression='gzip')
        print(f"  Features have been saved to: {h5_features_path}")
    
    with h5py.File(h5_labels_path, 'w') as f:
        f.create_dataset('labels', data=all_labels, compression='gzip')
        print(f"  The labels has been saved to: {h5_labels_path}")
    
    return all_features, all_labels

def check_all_trk_files_exist(root_path, expected_tracts):
    
    for trk_name in expected_tracts:
        trk_path = os.path.join(root_path, trk_name + ".trk")
        if not os.path.exists(trk_path):
            return False
    return True

def process_all_samples(root_dir, expected_tracts, output_dir, output_json_dir=None):
    """
    Process all sample folders
    Parameters:
        root_dir: Root directory containing multiple sample folders
        expected_tracts: List of expected tract names
        output_dir: Output directory
        output_jason_dir: Directory for saving JSON files
    """
    # Create a global label mapping (where each sample uses the same mapping)
    label_mapping = OrderedDict()
    for i, tract_name in enumerate(expected_tracts, 1):
        label_mapping[tract_name] = i
    
    print(f"Global label Mapping:")
    for tract_name, label in label_mapping.items():
        print(f"  {tract_name} -> label {label}")
    
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
        mapping_path = os.path.join(output_json_dir, "global_label_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    
    all_sample_list = sorted(os.listdir(root_dir))
    for modal in ["train", "eval", "test"]:
        if modal == "train":
            sample_list = all_sample_list[:int(0.6*len(all_sample_list))]
        elif modal == "eval":
            sample_list = all_sample_list[int(0.6*len(all_sample_list)):int(0.8*len(all_sample_list))]
        elif modal == "test":
            sample_list = all_sample_list[int(0.8*len(all_sample_list)):]

        for item in sample_list:
            item_path = os.path.join(root_dir, item, "tracts")
            
            if not os.path.isdir(item_path):
                continue
            
            all_trk_exist = check_all_trk_files_exist(item_path, expected_tracts)
            
            if not all_trk_exist:
                continue
            
            out_sample_dir = os.path.join(output_dir, modal, item)
            
            features, labels = process_sample(
                item_path, 
                expected_tracts, 
                label_mapping, 
                out_sample_dir,
                num_points=15
            )
            
    print(f"\n{'='*80}")
    print("All samples have been processed.!")
    print(f"{'='*80}")
    
    
if __name__ == "__main__":
    expected_tracts = [
        'UF_right', 'UF_left',    # Uncinate Fasciculus
        'AF_right', 'AF_left',    # Arcuate Fasciculus
        'IFO_right', 'IFO_left',  # Inferior Fronto-Occipital
        'SLF_III_right', 'SLF_III_left'   # Superior Longitudinal Fasciculus
    ]
    root_dir = ""  
    output_json_dir = ""
    output_dir = ""
    process_all_samples(root_dir, expected_tracts, output_dir, output_json_dir)
