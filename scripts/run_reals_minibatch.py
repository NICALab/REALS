'''
Run this after generating samples from generate_oscillation.py!!!
python -m scripts.run_reals_minibatch
'''
import os
import numpy as np
import torch
import skimage.io as skio
import scipy.io as scio
import argparse
from reals.REALS_minibatch import REALS_minibatch

parser = argparse.ArgumentParser()
parser.add_argument('--lr_wwt', type=float, default=1e-4, help='learning rate of W.')
parser.add_argument('--lr_tau', type=float, default=1e-3, help='learning rate of tau.')
parser.add_argument('--epoch', type=int, default=2000, help='number of iterations.')
parser.add_argument('--batch_size', type=int, default=None, help='batch size, for mini-batch training.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--rank', type=int, default=1, help='rank of W')
parser.add_argument('--tau', type=str, default='euclidean', help='transformation to use. affine, euclidean are avail.')
parser.add_argument('--verbose', type=bool, default=True, help='Print stuffs. (Time/Epoch/loss)')
args = parser.parse_args()


def learn_tau_reals(Y, args, tau_3x3):
    Y = Y.permute(1, 2, 0)  # 512x512x60
    Y /= Y.max()
    data_shape = list(Y.size())  # 512x512x60
    Y_res = Y.reshape(np.prod(data_shape[0:2]), data_shape[2]).permute(1, 0)  # 60x(512x512)

    [L_res, S_res, L_stat_res, _, model, elapsed_time] = REALS_minibatch(Y_res, args, data_shape)

    with torch.no_grad():
        if args.tau == 'affine':
            tau_bear = model.theta
        elif args.tau == 'euclidean':
            tau_bear = torch.zeros((data_shape[-1], 2, 3))
            tau_bear[:, 0, 0] = torch.cos(model.theta_ro[:, 0])
            tau_bear[:, 1, 1] = torch.cos(model.theta_ro[:, 0])
            tau_bear[:, 0, 1] = -torch.sin(model.theta_ro[:, 0])
            tau_bear[:, 1, 0] = torch.sin(model.theta_ro[:, 0])
            tau_bear[:, 0, 2] = model.theta_tr[:, 0]
            tau_bear[:, 1, 2] = model.theta_tr[:, 1]
        else:
            exit()
        tau_bear_3x3 = torch.zeros((data_shape[-1], 3, 3))
        tau_bear_3x3[:, 0:2, :] = tau_bear
        tau_bear_3x3[:, 2, 2] = 1
        tau_bear_comp_3x3 = torch.matmul(tau_3x3, tau_bear_3x3)

        L = L_res.permute(1, 0).reshape(data_shape)
        S = S_res.permute(1, 0).reshape(data_shape)
        L_stat = L_stat_res.permute(1, 0).reshape(data_shape)

        if data_shape[0] >= data_shape[1]:
            Y_np = Y.permute(2, 0, 1).numpy()
            L_np = L.permute(2, 0, 1).numpy()
            S_np = S.permute(2, 0, 1).numpy()
            L_stat_np = L_stat.permute(2, 0, 1).numpy()
        else:
            Y_np = Y.permute(2, 1, 0).numpy()
            L_np = L.permute(2, 1, 0).numpy()
            S_np = S.permute(2, 1, 0).numpy()
            L_stat_np = L_stat.permute(2, 1, 0).numpy()

    return Y_np, L_np, L_stat_np, S_np, tau_bear_comp_3x3, elapsed_time


def main(args):
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/batch"):
        os.mkdir("./results/batch")

    folder_list = ['./data/2d_zebrafish_brain_data/normal/tr_6.0_ro_4.0']
    batch_list = [5, 15, 30, 60]

    for folder in folder_list:
        for batch_size in batch_list:
            args.batch_size = batch_size
            root = folder.split('/')[-1]  # tr_0.0_ro_0.0
            result_folder = f'batch/reals_batch_{batch_size}_{root}'
            print('current file: ', result_folder)
            if not os.path.exists(f"./results/{result_folder}"):
                os.mkdir(f"./results/{result_folder}")
            if not os.path.exists(f"./results/{result_folder}/Y_{root}"):
                os.mkdir(f"./results/{result_folder}/Y_{root}")
            if not os.path.exists(f"./results/{result_folder}/L_{root}"):
                os.mkdir(f"./results/{result_folder}/L_{root}")
            if not os.path.exists(f"./results/{result_folder}/S_{root}"):
                os.mkdir(f"./results/{result_folder}/S_{root}")
            if not os.path.exists(f"./results/{result_folder}/L_stat_{root}"):
                os.mkdir(f"./results/{result_folder}/L_stat_{root}")

            for i in range(1):
                Y = torch.from_numpy(skio.imread(f'{folder}/Y_{root}_{i}.tif').astype(float)).float()
                tau_3x3 = torch.from_numpy(scio.loadmat(f'{folder}/tau_{root}_{i}.mat')['tau'])
                print(f'info about Y_{i} = size: {Y.size()}, max: {torch.max(Y)}, min: {torch.min(Y)}, type: {Y.dtype}')
                print(f'info about tau_{i} = tau[0]: {tau_3x3[0]}, type: {tau_3x3.dtype}')

                Y_np, L_np, L_stat_np, S_np, tau_bear_comp_3x3, elapsed_time = learn_tau_reals(Y.clone(), args, tau_3x3)

                Y_L_Lstat_S_np = np.concatenate((Y_np, L_np, L_stat_np, S_np), axis=2)
                skio.imsave(f"./results/{result_folder}/Y_{root}/Y_{root}_{i}.tif", Y_np)
                skio.imsave(f"./results/{result_folder}/L_{root}/L_{root}_{i}.tif", L_np)
                skio.imsave(f"./results/{result_folder}/S_{root}/S_{root}_{i}.tif", S_np)
                skio.imsave(f"./results/{result_folder}/L_stat_{root}/L_stat_{root}_{i}.tif", L_stat_np)
                skio.imsave(f"./results/{result_folder}/Y_L_Lstat_S_{root}_{i}.tif", Y_L_Lstat_S_np)

                tau_bear_comp = tau_bear_comp_3x3[:, 0:2, :]
                scio.savemat(f'./results/{result_folder}/tau_{i}.mat', {'tau': tau_bear_comp.numpy()})
                scio.savemat(f'./results/{result_folder}/time_{i}.mat', {'time': elapsed_time / 1000})  # second


if __name__ == "__main__":
    main(args)