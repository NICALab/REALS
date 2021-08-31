"""
Run this after downloading 3-D Zbrafish brain data!!!
python -m scripts.run_reals_3d
"""
import os
import numpy as np
import torch
import skimage.io as skio
import argparse
import time
from reals.REALS_3d import REALS_3d

parser = argparse.ArgumentParser()
parser.add_argument('--lr_wwt', type=float, default=1e-4, help='learning rate of W.')
parser.add_argument('--lr_tau', type=float, default=1e-3, help='learning rate of tau.')
parser.add_argument('--epoch', type=int, default=1000, help='number of iterations.')
parser.add_argument('--batch_size', type=int, default=10, help='batch size, for mini-batch training.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--rank', type=int, default=1, help='rank of W')
parser.add_argument('--tau', type=str, default='affine', help='transformation to use. affine, euclidean are avail.')
parser.add_argument('--verbose', type=bool, default=True, help='Print stuffs. (Time/Epoch/loss)')
parser.add_argument("--dir", default='./data/3d_zebrafish_brain_data', type=str, help="directory of the tif")
parser.add_argument("--type", default='normal', type=str, help="type of folder. normal/br/no.")
args = parser.parse_args()


def main(args):
    """
    Data description
    Y          : torch.Tensor; [x, y, t] or [x, y, z, t]
    Y_res      : torch.Tensor; [t, xy]   or [t, xyz] (reshape and transpose)
    data_shape : list; [x, y, t] or [x, y, z, t]

    [L_res, S_res, total_loss, model, time] = train_test_BEAR(Y_res, config)
    L_res      : torch.Tensor; [t, xy]   or [t, xyz]
    L          : torch.Tensor; [x, y, t] or [x, y, z, t] (transpose and reshape)
    S_res      : torch.Tensor; [t, xy]   or [t, xyz]
    S          : torch.Tensor; [x, y, t] or [x, y, z, t] (transpose and reshape)
    """
    folder = f'{args.dir}'
    Y = torch.from_numpy(skio.imread(f'{folder}/Y_tr_4_ro_2.tif').astype(float)).float()  # t,w,h,d
    Y = Y.permute(1, 2, 3, 0)  # w,h,d,t
    Y /= Y.max()
    data_shape = list(Y.size())  # (w,h,d,t)
    Y_res = Y.reshape(np.prod(data_shape[0:3]), data_shape[3]).permute(1, 0)  # t,whd

    print(f"""
    START TIME : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    T_XY SHAPE : {Y_res.size()}
    CONFIG     : {args}""")

    [L_res, S_res, L_stat_res, total_loss, _, time_] = REALS_3d(Y_res, args, data_shape)
    Y = Y_res.permute(1, 0).reshape(data_shape)
    del Y_res

    L = L_res.permute(1, 0).reshape(data_shape)
    S = S_res.permute(1, 0).reshape(data_shape)
    L_stat = L_stat_res.permute(1, 0).reshape(data_shape)

    print(f"""Finished. Elapsed time : {time_}, Total Loss : {total_loss:.3f}""")

    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists(f"./results/3d_zebrafish"):
        os.mkdir(f"./results/3d_zebrafish")
    if not os.path.exists(f"./results/3d_zebrafish/Y"):
        os.mkdir(f"./results/3d_zebrafish/Y")
    if not os.path.exists(f"./results/3d_zebrafish/L"):
        os.mkdir(f"./results/3d_zebrafish/L")
    if not os.path.exists(f"./results/3d_zebrafish/S"):
        os.mkdir(f"./results/3d_zebrafish/S")
    if not os.path.exists(f"./results/3d_zebrafish/L_stat"):
        os.mkdir(f"./results/3d_zebrafish/L_stat")

    result_dict = {'Y': Y, 'L': L, 'L_stat': L_stat, 'S': S}
    proj_x, proj_y, proj_z = [], [], []
    for key in result_dict:
        sample = result_dict[key]
        sample = sample.permute(3, 0, 1, 2).numpy()  # t,w,h,d
        proj_x_sample = np.max(sample, axis=1)
        proj_y_sample = np.max(sample, axis=2)
        proj_z_sample = np.max(sample, axis=3)
        proj_x.append(proj_x_sample)
        proj_y.append(proj_y_sample)
        proj_z.append(proj_z_sample)
        skio.imsave(f"./results/3d_zebrafish/{key}/{key}.tif", sample)
        skio.imsave(f"./results/3d_zebrafish/{key}/{key}_proj_x.tif", proj_x_sample)
        skio.imsave(f"./results/3d_zebrafish/{key}/{key}_proj_y.tif", proj_y_sample)
        skio.imsave(f"./results/3d_zebrafish/{key}/{key}_proj_z.tif", proj_z_sample)

    Y_L_Lstat_S_proj_x = np.concatenate(proj_x, axis=2)
    Y_L_Lstat_S_proj_y = np.concatenate(proj_y, axis=2)
    Y_L_Lstat_S_proj_z = np.concatenate(proj_z, axis=1)
    Y_L_Lstat_S_proj_x = Y_L_Lstat_S_proj_x.transpose((0, 2, 1))
    Y_L_Lstat_S_proj_y = Y_L_Lstat_S_proj_y.transpose((0, 2, 1))
    Y_L_Lstat_S_proj_z = Y_L_Lstat_S_proj_z.transpose((0, 2, 1))
    skio.imsave(f"./results/3d_zebrafish/Y_L_Lstat_S_proj_x.tif", Y_L_Lstat_S_proj_x)
    skio.imsave(f"./results/3d_zebrafish/Y_L_Lstat_S_proj_y.tif", Y_L_Lstat_S_proj_y)
    skio.imsave(f"./results/3d_zebrafish/Y_L_Lstat_S_proj_z.tif", Y_L_Lstat_S_proj_z)


if __name__ == "__main__":
    main(args)