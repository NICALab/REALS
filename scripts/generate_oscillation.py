'''
This code generates xy translated and rotated samples from static Y.
python -m scripts.generate_oscillation
'''
import os
import numpy as np
import torch
import torch.nn.functional as F
import skimage.io as skio
import scipy.io as scio
torch.manual_seed(0)
np.random.seed(0)


def generate_tau(data_shape, rand_noise, tr, ro):
    t, w, h = data_shape
    tau = torch.zeros((t, 2, 3))
    tau[:, 0, 2] = tr * (2 / h) * rand_noise[0]  # affine_grid range [-1, 1]
    tau[:, 1, 2] = tr * (2 / w) * rand_noise[1]
    rand_angle = ro * rand_noise[2]
    tau[:, 0, 0] = torch.cos(rand_angle)
    tau[:, 1, 1] = torch.cos(rand_angle)
    tau[:, 0, 1] = -torch.sin(rand_angle)
    tau[:, 1, 0] = torch.sin(rand_angle)
    return tau


def main():
    dir = './data/2d_zebrafish_brain_data'
    path = f'{dir}/Y.tif'
    translation_level = np.arange(0, 30, 3, dtype=np.float32)
    rotation_level = np.arange(0, 20, 2, dtype=np.float32) * (np.pi * 1 / 180)
    n_samples = 5

    if not os.path.exists(f"{dir}/normal"):
        os.mkdir(f"{dir}/normal")

    for tr in translation_level:
        for ro in rotation_level:
            ro_rad = ro * (180 / np.pi)
            for i in range(n_samples):
                subdir = f"tr_{np.round(tr)}_ro_{np.round(ro_rad)}"
                if not os.path.exists(f"{dir}/normal/{subdir}"):
                    os.mkdir(f"{dir}/normal/{subdir}")

                Y = torch.from_numpy(skio.imread(path).astype(float)).float()[:512]  # (t,w,h)
                t, w, h = Y.size()
                rand_noise = 2 * torch.rand((3, t)) - 1
                tau = generate_tau(Y.size(), rand_noise, tr, ro)

                tau_3x3 = torch.zeros((t, 3, 3))
                tau_3x3[:, 0:2, :] = tau
                tau_3x3[:, 2, 2] = 1

                # affine transformation
                Y_reshape = Y.view(t, 1, w, h)
                grid = F.affine_grid(tau, Y_reshape.size())
                Y_reg = F.grid_sample(Y_reshape, grid)[:, 0, ...]  # (t,w,h)

                # save the result
                skio.imsave(f"{dir}/normal/{subdir}/Y_tr_{np.round(tr)}_ro_{np.round(ro_rad)}_{i}.tif", Y_reg.numpy())
                scio.savemat(f"{dir}/normal/{subdir}/tau_tr_{np.round(tr)}_ro_{np.round(ro_rad)}_{i}.mat", {'tau': tau_3x3.numpy()})


if __name__=="__main__":
    main()