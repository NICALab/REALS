# Robust and Efficient Image Alignment through Simultaneous Low Rank and Sparse Decomposition

This repository is the official implementation of [Robust and Efficient Image Alignment through Simultaneous Low Rank and Sparse Decomposition](https://arxiv.org/abs/2030.12345).

## Requirements

To install requirements:

```setup
conda create -n reals python=3.8.10
conda activate reals
pip install -r requirements.txt
```

Data for REALS is required for execution of the code. 
- [2-D Zebrafish brain data](https://drive.google.com/drive/folders/19jtKft4TbDQj4SMBEcy13drACZGmNDeG?usp=sharing): Static data with size [512(x), 512(y), 60(z)]. Additional perturbations or noise can be applied.
- [3-D Zebrafish brain data](https://drive.google.com/drive/folders/1sPP-dAnScBAyF-3K_W0CfBJzuO4cEXhB?usp=sharing): Data with size [256(x), 512(y), 48(z), 60(t)]. Random translation and rotation are applied synthetically.
- [Mouse brain data](https://drive.google.com/drive/folders/1ZW4MOOapABmX4Q3C0Is_HMkf7DbORgHU?usp=sharing): Data with size [480(x), 752(y), 1000(t)]. `Y.tif` is the original video, and `Y_DoG.tif` is high pass filtered video.
- [2-D Zebrafish brain data with drift](https://drive.google.com/drive/folders/1P1CLvrtF07NHrw8FRIGKWrbexccTI7FV?usp=sharing): Data with size [256(x), 512(y), 600(t)]. Original video has small drift.

To run `run_reals.py`, you have to generate data with perturbation from static 2-D Zebrafish brain data. 
- `generate_oscillation.py` generates randomly perturbed data with corresponding transformation from static 2-D Zebrafish brain data.

If you want to add noise to above generated data,
- `generate_gaussian_noise.py` generates data with gaussian noise from perturbed data.
- `generate_poisson_noise.py` generates data with poisson noise from perturbed data.

## Training

`run_reals.py` can reproduce most of the results in the paper. Run following command:
```run_reals
python -m scripts.run_reals --type normal
```
To perform REALS on 3-D Zebrafish brain data, run following command: 
```run_reals_3d
python -m scripts.run_reals_3d
```
We also provide the mini-batch and multi-resolution version of REALS in `run_reals_minibatch.py`, `run_reals_multi.py` respectively.

## Evaluation

We provide 3 sample codes for evaluation. However, the code may require modifications, which depends on the results. 
- `eval_heatmap.py` shows the alignment inconsistency heatmap.  
- `eval_barplot.py` shows the alignment inconsistency bar plot of REALS. It is to check performance with different batch size.
- `eval_timeplot.py` shows the (time, alignment inconsistency) plot. You need to save time and transformation at each iteration to plot the result.

## Results


## Contributing
The base code is from [BEAR](https://github.com/NICALab/BEAR). We will make it public after the review.
