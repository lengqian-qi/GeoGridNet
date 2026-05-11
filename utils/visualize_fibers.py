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

def plot_fibers_with_labels(features_h5_path, labels_h5_path, expected_tracts, output_path="fiber_visualization.png"):
    """
    Plot all fibre bundles, distinguishing them with eight different colours according to labels, against a black background.
    Parameters:
        features_h5_path: Feature file path, shape [N, 15, 3]
        labels_h5_path: Label file path, shape [N,]
        output_path: Output image path
    """
    
    with h5py.File(features_h5_path, 'r') as f:
        features = f['features'][:]
    
    with h5py.File(labels_h5_path, 'r') as f:
        labels = f['labels'][:]
    
    colors = plt.cm.tab20c(np.linspace(0, 1, 8))

    fig = plt.figure(figsize=(10, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    ax.axis('off')
    ax.grid(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    
    ax.view_init(elev=20, azim=45)
    
    unique_labels = np.unique(labels)
    
    for label_idx in unique_labels:
        label = int(label_idx)
        
        fiber_indices = np.where(labels == label)[0]
        
        if len(fiber_indices) == 0:
            continue
            
        color = colors[label - 1]  
        
        for idx in fiber_indices:
            fiber = features[idx]  # [15, 3]
            
            ax.plot(fiber[:, 0], fiber[:, 1], fiber[:, 2], 
                   color=color, alpha=0.3, linewidth=0.5)
            # ax.scatter(
            #     fiber[:, 0],
            #     fiber[:, 1],
            #     fiber[:, 2],
            #     color=color,
            #     s=0.8,          #
            #     alpha=0.6
            # )
            
    all_coords = features.reshape(-1, 3)
    
    center_x = (all_coords[:, 0].min() + all_coords[:, 0].max()) / 2
    center_y = (all_coords[:, 1].min() + all_coords[:, 1].max()) / 2
    center_z = (all_coords[:, 2].min() + all_coords[:, 2].max()) / 2
    
    max_range = max(
        all_coords[:, 0].max() - all_coords[:, 0].min(),
        all_coords[:, 1].max() - all_coords[:, 1].min(),
        all_coords[:, 2].max() - all_coords[:, 2].min()
    )
    
    ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
    ax.set_ylim(center_y - max_range/2, center_y + max_range/2)
    ax.set_zlim(center_z - max_range/2, center_z + max_range/2)
    
    legend_elements = []
    for label in unique_labels:
        color = colors[int(label) - 1]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                         label=f'{expected_tracts[label-1]}'))
    
    legend = ax.legend(handles=legend_elements, 
                      loc='upper left',
                      fontsize=6,
                      framealpha=0.3)
    
    for text in legend.get_texts():
        text.set_color('white')
    
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    
    plt.tight_layout()
    
    plt.savefig(output_path, 
                dpi=300, 
                facecolor='black', 
                edgecolor='none',
                bbox_inches='tight',
                pad_inches=0.1)
    
    print(f"The image has been saved to: {output_path}")
    
    return features, labels

    
if __name__ == "__main__":
    expected_tracts = [
        'UF_right', 'UF_left',    # Uncinate Fasciculus
        'AF_right', 'AF_left',    # Arcuate Fasciculus
        'IFO_right', 'IFO_left',  # Inferior Fronto-Occipital
        'SLF_III_right', 'SLF_III_left'   # Superior Longitudinal Fasciculus
    ]

    features_h5_path = ""
    labels_h5_path = ""
    plot_fibers_with_labels(features_h5_path, labels_h5_path, expected_tracts, output_path="fiber_visualization.png")