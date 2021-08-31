'''
python -m scripts.eval_heatmap
'''
import glob
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


def plot_heatmap(all_method_type):
    all_method_result = []
    for method, method_heatmap in all_method_type.items():
        all_method_result.append(method_heatmap)
    all_method_result_np = np.array(all_method_result)
    best_heatmap = np.argmin(all_method_result_np, axis=0)

    for i in range(4):
        plt.matshow(all_method_type[all_method_names[i]])
        plt.gca().invert_yaxis()
        plt.colorbar(shrink=0.8, aspect=10)
        plt.show()
        plt.close()

    len_y = best_heatmap.shape[0]
    len_x = best_heatmap.shape[1]
    board = np.zeros((len_y, len_x, 3))
    color = [[125, 46, 141], [0, 113, 188], [216, 82, 24], [236, 176, 31]]
    color_np = np.array(color) / 255

    for i in range(len_y):
        for j in range(len_x):
            board[i, j] = color_np[best_heatmap[i, j]]

    plt.imshow(board)
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()


def calculate_tau_std(tau):
    t, _, _ = tau.shape
    tau = tau.reshape((t, -1))  # tx6
    tau_mu = np.tile(np.mean(tau, axis=0, keepdims=True), (t, 1))  # tx6
    tau_std = np.linalg.norm(tau - tau_mu, ord=1, axis=1)
    return tau_std


'''
normal = (xy tr, ro) 
noise_p = (xy tr ro, br)
noise_g = (xy tr ro, no)
df = (xy tr ro, L)
'''
option = 'normal'
if option == 'normal':
    indicator = 'tr_*_ro_*'
    mapping_x = {'0.0': 0, '3.0': 1, '6.0': 2, '9.0': 3, '12.0': 4, '15.0': 5,
                 '18.0': 6, '21.0': 7, '24.0': 8, '27.0': 9}  # tr
    mapping_y = {'0.0': 0, '2.0': 1, '4.0': 2, '6.0': 3, '8.0': 4, '10.0': 5,
                 '12.0': 6, '14.0': 7, '16.0': 8, '18.0': 9}  # ro
elif option == 'br':
    indicator = 'br_*_tr_*_ro_*'
    mapping_x = {'1000.0': 1, '333.0': 2, '100.0': 3, '33.0': 4}  # br
    mapping_y = {'0.0': 0, '6.0': 1, '12.0': 2, '18.0': 3, '24.0': 4}  # tr
elif option == 'no':
    indicator = 'no_*_tr_*_ro_*'
    mapping_x = {'0.1': 1, '0.3': 2, '1.0': 3, '3.0': 4}  # no
    mapping_y = {'0.0': 0, '6.0': 1, '12.0': 2, '18.0': 3, '24.0': 4}  # tr
elif option == 'df':
    indicator = 'L_*_tr_*_ro_*'
    mapping_x = {'0.8': 1, '0.6': 2, '0.4': 3, '0.2': 4}  # L intensity
    mapping_y = {'0.0': 0, '6.0': 1, '12.0': 2, '18.0': 3, '24.0': 4}  # tr
else:
    exit()

if option == 'normal':
    len_x = len(mapping_x)
else:
    len_x = len(mapping_x) + 1
len_y = len(mapping_y)

all_method_names = ['bear', 't_grasta', 'rasl', 'sralt']
all_method_folders = {'bear': glob.glob(f'./results/bear_{indicator}'),
                      't_grasta': glob.glob(f'./results/t_grasta_{indicator}'),
                      'rasl': glob.glob(f'./results/rasl_{indicator}'),
                      'sralt': glob.glob(f'./results/sralt_{indicator}')}

all_method_measure = {'bear': np.zeros((len_y, len_x), np.float32),
                      't_grasta': np.ones((len_y, len_x), np.float32),
                      'rasl': np.ones((len_y, len_x), np.float32),
                      'sralt': np.ones((len_y, len_x), np.float32)}
all_method_time = {'bear': np.ones((len_y, len_x), np.float32),
                   't_grasta': np.ones((len_y, len_x), np.float32),
                   'rasl': np.ones((len_y, len_x), np.float32),
                   'sralt': np.ones((len_y, len_x), np.float32)}


if not option == 'normal':
    for method, method_folders in all_method_measure.items():
        for tr, index in mapping_y.items():
            folder = f'./results/{method}_tr_{tr}_ro_{float(tr) * (2/3)}'
            method_tau_files = []
            method_time_files = []
            for i in range(5):
                method_tau_files = method_tau_files + glob.glob(f'{folder}/tau_{i}.mat')
                method_time_files = method_time_files + glob.glob(f'{folder}/time_{i}.mat')

            if len(method_tau_files) == 0:
                continue

            tau_method_std_list = []
            time_method_list = []

            for tau_f, time_f in zip(method_tau_files, method_time_files):
                tau_method = scio.loadmat(tau_f)['tau']
                time_method = scio.loadmat(time_f)['time']
                tau_method_std_list.append(calculate_tau_std(tau_method))
                time_method_list.append(time_method)

            tau_method_std_np = np.concatenate(tau_method_std_list)
            measure = np.mean(tau_method_std_np)
            avg_time = np.mean(time_method_list)
            all_method_measure[method][index, 0] = measure
            all_method_time[method][index, 0] = avg_time


for method, method_folders in all_method_folders.items():
    print(method)
    print(method_folders)

    for folder in method_folders:
        method_tau_files = []
        method_time_files = []
        for i in range(5):
            method_tau_files = method_tau_files + glob.glob(f'{folder}/tau_{i}.mat')
            method_time_files = method_time_files + glob.glob(f'{folder}/time_{i}.mat')

        if len(method_tau_files) == 0:
            continue

        tau_method_std_list = []
        time_method_list = []
        token = (folder.split('/')[-1]).split('_')
        if method == 't_grasta':
            tr = mapping_x[token[3]]
            ro = mapping_y[token[5]]
        else:
            tr = mapping_x[token[2]]
            ro = mapping_y[token[4]]

        for tau_f, time_f in zip(method_tau_files, method_time_files):
            tau_method = scio.loadmat(tau_f)['tau']
            time_method = scio.loadmat(time_f)['time']
            tau_method_std_list.append(calculate_tau_std(tau_method))
            time_method_list.append(time_method)
            print(f'{method}_{tau_f}_{np.mean(calculate_tau_std(tau_method))}')

        tau_method_std_np = np.concatenate(tau_method_std_list)
        measure = np.mean(tau_method_std_np)
        avg_time = np.mean(time_method_list)   # second
        all_method_measure[method][ro, tr] = measure
        all_method_time[method][ro, tr] = avg_time


plot_heatmap(all_method_measure)
plot_heatmap(all_method_time)
