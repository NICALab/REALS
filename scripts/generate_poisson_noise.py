'''
This code generates poisson noise data from oscillated data from generate_oscillation.py
python -m scripts.generate_poisson_noise
'''
import os
import glob
import numpy as np
import torch
import skimage.io as skio
import scipy.io as scio
torch.manual_seed(0)
np.random.seed(0)

def main():
    dir = './data/2d_zebrafish_brain_data'
    noise_list = [(0.0, 0.0), (6.0, 4.0), (12.0, 8.0), (18.0, 12.0), (24.0, 16.0)]
    brightness_level = torch.from_numpy(np.array([1000, 333, 100, 33], np.float32))
    n_samples = 5

    if not os.path.exists(f"{dir}/br"):
        os.mkdir(f"{dir}/br")

    folder_list = []
    for tr, ro in noise_list:
        folder_list = folder_list + glob.glob(f'{dir}/normal/tr_{tr}_ro_{ro}')

    for folder in folder_list:
        root = folder.split('/')[-1]  # tr_0.0_ro_0.0
        tr, ro = root.split('_')[1], root.split('_')[3]
        for br in brightness_level:
            subdir = f"br_{br}_tr_{tr}_ro_{ro}"
            if not os.path.exists(f"{dir}/br/{subdir}"):
                os.mkdir(f"{dir}/br/{subdir}")

            for i in range(n_samples):
                Y_reg = torch.from_numpy(skio.imread(f'{folder}/Y_{root}_{i}.tif').astype(float)).float()  # (t,w,h)
                tau_3x3 = torch.from_numpy(scio.loadmat(f'{folder}/tau_{root}_{i}.mat')['tau'])
                max_Y_reg = torch.max(Y_reg)
                Y_reg_p = torch.round(br * Y_reg / max_Y_reg).type(torch.int16)
                Y_reg_p = torch.from_numpy(np.random.poisson(Y_reg_p)).type(torch.int16)
                Y_reg_p = Y_reg_p.type(torch.float32) / br * max_Y_reg
                Y_reg_p = torch.max(Y_reg_p, torch.tensor([0.]))  # image is positive
                Y_reg_p = torch.min(Y_reg_p, torch.tensor([max_Y_reg]))  # thresholding large values
                # save the result
                skio.imsave(f"{dir}/br/{subdir}/Y_br_{br}_tr_{tr}_ro_{ro}_{i}.tif", Y_reg_p.numpy())
                scio.savemat(f"{dir}/br/{subdir}/tau_br_{br}_tr_{tr}_ro_{ro}_{i}.mat", {'tau': tau_3x3.numpy()})


if __name__=="__main__":
    main()