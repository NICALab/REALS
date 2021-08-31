'''
python -m scripts.eval_timeplot
'''
import glob
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


def calculate_tau_std(tau):
    t, _, _ = tau.shape
    tau = tau.reshape((t, -1))  # tx6
    tau_mu = np.tile(np.mean(tau, axis=0, keepdims=True), (t, 1))  # tx6
    tau_std = np.linalg.norm(tau - tau_mu, ord=1, axis=1)
    return tau_std


tr_ro_list = [('12.0', '8.0'), ('18.0', '12.0'), ('24.0', '16.0')]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
legend = []
use_iter = False

for tr, ro in tr_ro_list:
    result_folder_list = [f'./eval_time/0_time_ai_tr_{tr}_ro_{ro}',
                          f'./eval_time/rasl_tr_{tr}_ro_{ro}',
                          f'./eval_time/t_grasta_tr_{tr}_ro_{ro}',
                          f'./eval_time/sralt_tr_{tr}_ro_{ro}',]
    init_tau_path = f'./data/210322_Casper_GCaMP7a_4dpfrawData210322_Confocal_16X0p8NA_water_Casper_GCaMP7a_4dpf_FITC_' \
                    f'scanSpeed_1Hz_fastTimelapse_SinglePlane_sample1_z8/normal/tr_{tr}_ro_{ro}/tau_tr_{tr}_ro_{ro}_0.mat'

    tau_method = scio.loadmat(init_tau_path)['tau']
    init_tau_y = np.mean(calculate_tau_std(tau_method))
    print(init_tau_y)

    for result_folder in result_folder_list:
        if result_folder == f'./eval_time/0_time_ai_tr_{tr}_ro_{ro}':
            indicator = '*_tau.mat'
            indicator_ = '*_time.mat'
        else:
            indicator = '*_tau_inner.mat'
            indicator_ = '*_time_inner.mat'

        method_tau_files = sorted(glob.glob(f'./results/{result_folder}/{indicator}'),
                                  key=lambda x: int((x.split('/')[-1]).split('_')[0]))
        method_time_files = sorted(glob.glob(f'./results/{result_folder}/{indicator_}'),
                                   key=lambda x: int((x.split('/')[-1]).split('_')[0]))
        print(method_tau_files[:10])
        if len(method_time_files) == 0:
            continue
        else:
            x = [0.0]
            y = [init_tau_y]
            for tau_f, time_f in zip(method_tau_files, method_time_files):
                tau_method = scio.loadmat(tau_f)['tau']
                time_method = scio.loadmat(time_f)['time']
                y.append(np.mean(calculate_tau_std(tau_method)))
                x.append(time_method.flatten())
            # ax.plot(np.arange(len(y)), y)
            if use_iter:
                if result_folder == f'./eval_time/rasl_tr_{tr}_ro_{ro}':
                    min_index = np.argmin(y)
                    ax.plot(np.arange(len(y[:min_index])), y[:min_index])
                else:
                    ax.plot(np.arange(len(y)), y)
                legend.append(result_folder)
            else:
                if result_folder == f'./eval_time/rasl_tr_{tr}_ro_{ro}':
                    min_index = np.argmin(y)
                    ax.plot(x[:min_index], y[:min_index])
                else:
                    ax.plot(x, y)
                legend.append(result_folder)

ax.set_yscale('log')
ax.set_xscale('symlog')
plt.xlim([0.0, 100000])
plt.legend(legend)
plt.show()