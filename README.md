# GeogridNet
Official implementation of "GeogridNet: A Dual-Branch Geometric Grid Network for Superior Temporal Sulcus-Related Fiber Parcellation"

![model](Framework.png)

>Tractography parcellation of superior temporal sulcus (STS) related fibers is essential for investigating the white matter alterations of different fiber bundles in neurodegenerative diseases like AD. The core of tractography parcellation lies in classifying fiber streamlines into anatomically consistent bundles. Current tractography parcellation methods mainly model streamlines as point cloud data. However, existing research primarily considers the permutation invariance of streamlines while neglecting their spatial continuity and geometric consistency. This limitation restricts the parcellation performance for fiber bundles in the STS region. To address this issue, this study proposes a dual-branch network named Geometric Grid Network (GeoGridNet). Specifically, we introduce an enhanced point cloud branch GeometricPointNet, with ArcLength Positional Encoding and Geometric Attention mechanism, which explicitly models the geometric properties of fiber streamlines. Moreover, an Adaptive Spatial Grid Mapping (ASGM) module is devised to transform the streamlines into multi-view 2D grid representations. Finally, by fusing the features from both branches, our model achieves coordinated modeling with geometric consistency and spatial continuity in fibers. Experiments on public HCP and in-house CLAS datasets demonstrate that the proposed GeoGridNet outperforms existing SOTA tractography parcellation models quantitatively and qualitatively, validating the effectiveness of the presented architecture.

## Requirements
```bash
conda env create -f environment.yml
```

## Dataset
- Human Connectome Project (HCP) dataset
- China Longitudinal Aging Study (CLAS) dataset

The HCP dataset can be downloaded in [HCP](https://humanconnectome.org/study/hcp-young-adult/data-use-terms).

## Training 
We use `dataset/extract_tract_feat_trk.py` to convert fiber streamline `.trk` files into `.h5` format point cloud features.
After conversion, the dataset root directory should be organized as follows:

```dataset
dataset/
├── train/
  ├──sample1/
    ├──feature.h5
    ├──label.h5
  ├──sample2/
    ├──feature.h5
    ├──label.h5
  ...
├── val/
└── test/
```

Run the following command to start training:
```bash
python train.py --experiment_name GeoGridNet --device 0 --num_classes 8
```
## Testing
```bash
python predict.py --experiment_name GeoGridNet --device 0 --num_classes 8
```

