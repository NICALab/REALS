import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io as scio


def calculate_tau_std(tau):
    t, _, _ = tau.shape
    tau = tau.reshape((t, -1))  # tx6
    tau_mu = np.tile(np.mean(tau, axis=0, keepdims=True), (t, 1))  # tx6
    tau_std = np.linalg.norm(tau - tau_mu, ord=1, axis=1)
    return tau_std


x_map = {'batch_5': np.arange(0, 20, 5),
         'batch_15': np.arange(1, 20, 5),
         'batch_30': np.arange(2, 20, 5),
         'batch_60': np.arange(3, 20, 5)}

y_map = {'batch_5': np.zeros((4,)),
         'batch_15': np.zeros((4,)),
         'batch_30': np.zeros((4,)),
         'batch_60': np.zeros((4,))}

root = './results/batch'
path_list = ['reals_batch_*_tr_6.0_ro_4.0',
             'reals_batch_*_tr_12.0_ro_8.0',
             'reals_batch_*_tr_18.0_ro_12.0',
             'reals_batch_*_tr_24.0_ro_16.0']

for index, path in enumerate(path_list):
    folder_list = glob.glob(f'{root}/{path}')
    print(folder_list)
    for folder in folder_list:
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
        batch_size = token[2]
        print(batch_size)
        for tau_f, time_f in zip(method_tau_files, method_time_files):
            tau_method = scio.loadmat(tau_f)['tau']
            time_method = scio.loadmat(time_f)['time']
            tau_method_std_list.append(calculate_tau_std(tau_method))
            time_method_list.append(time_method)

        tau_method_std_np = np.concatenate(tau_method_std_list)
        measure = np.mean(tau_method_std_np)
        y_map[f'batch_{batch_size}'][index] = measure

ax = plt.subplot(111)
ax.bar(x_map['batch_5'], y_map['batch_5'], color='b', align='center')
ax.bar(x_map['batch_15'], y_map['batch_15'], color='r', align='center')
ax.bar(x_map['batch_30'], y_map['batch_30'], color='g', align='center')
ax.bar(x_map['batch_60'], y_map['batch_60'], color='black', align='center')
plt.show()
plt.close()