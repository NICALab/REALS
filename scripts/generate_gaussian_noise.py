'''
This code generates gaussian noise data from oscillated data from generate_oscillation.py
python -m scripts.generate_gaussian_noise
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
    noise_level = torch.tensor([0.1, 0.3, 1, 3])
    n_samples = 5

    if not os.path.exists(f"{dir}/no"):
        os.mkdir(f"{dir}/no")

    folder_list = []
    for tr, ro in noise_list:
        folder_list = folder_list + glob.glob(f'{dir}/normal/tr_{tr}_ro_{ro}')

    for folder in folder_list:
        root = folder.split('/')[-1]  # tr_0.0_ro_0.0
        tr, ro = root.split('_')[1], root.split('_')[3]
        for no in noise_level:
            subdir = f"no_{np.around(no.item(), decimals=1)}_tr_{tr}_ro_{ro}"
            if not os.path.exists(f"{dir}/no/{subdir}"):
                os.mkdir(f"{dir}/no/{subdir}")

            for i in range(n_samples):
                Y_reg = torch.from_numpy(skio.imread(f'{folder}/Y_{root}_{i}.tif').astype(float)).float()  # (t,w,h)
                tau_3x3 = torch.from_numpy(scio.loadmat(f'{folder}/tau_{root}_{i}.mat')['tau'])
                max_Y_reg = torch.max(Y_reg)
                Y_reg_n = Y_reg + max_Y_reg * (no / 100) * torch.randn_like(Y_reg)
                Y_reg_n = torch.max(Y_reg_n, torch.tensor([0.]))  # image is positive
                # save the result
                skio.imsave(f"{dir}/no/{subdir}/Y_no_{np.around(no.item(), decimals=1)}_tr_{tr}_ro_{ro}_{i}.tif", Y_reg_n.numpy())
                scio.savemat(f"{dir}/no/{subdir}/tau_no_{np.around(no.item(), decimals=1)}_tr_{tr}_ro_{ro}_{i}.mat", {'tau': tau_3x3.numpy()})


if __name__=="__main__":
    main()