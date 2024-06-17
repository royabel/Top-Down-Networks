import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
import statsmodels.stats.api as sms


matplotlib.use('TkAgg', force=True)


def analyze_task_sub_networks(analysis_dict):
    epochs = list(analysis_dict.keys())
    tasks = list(analysis_dict[epochs[0]].keys())
    layers = list(analysis_dict[epochs[0]][tasks[0]].keys())

    plt.rcParams['axes.grid'] = True
    fig, axs = plt.subplots(2, len(layers), sharey='row')

    xs = range(len(epochs))
    for l_i, l in enumerate(layers):
        axs[0, l_i].set_title(f'layer {l}')
        tasks_act = []
        for t in tasks:
            activations = [np.array(analysis_dict[e][t][l]).flatten() for e in epochs]
            tasks_act.append(activations)
            act_percent = [(x > 0).sum() / len(x) for x in activations]
            axs[0, l_i].plot(xs, act_percent, label=f'task {t}')

        b_cos_s = [cosine_similarity((tasks_act[0][i].reshape(1, -1) > 0), (tasks_act[1][i].reshape(1, -1) > 0))[0, 0] for i in
                   range(len(epochs))]
        axs[1, l_i].plot(xs, b_cos_s)


    axs[0, 0].set_ylabel('active percent')
    axs[1, 0].set_ylabel('binary cosine similarity')

    axs[0, 0].legend()

    plt.show()


def conf_int(a):
    interval_range = sms.DescrStatsW(a).tconfint_mean()
    return (interval_range[1] - interval_range[0]) / 2


def _generate_comparison_plot(results_dict, agg='mean', key_filter=None, exclude=None, const_val=None, start_idx=0):
    if key_filter is not None:
        if isinstance(key_filter, list):
            for kf in key_filter:
                results_dict = {k: v for k, v in results_dict.items() if kf in k}
        else:
            results_dict = {k: v for k, v in results_dict.items() if key_filter in k}

    if exclude is not None:
        if isinstance(exclude, list):
            for kf in exclude:
                results_dict = {k: v for k, v in results_dict.items() if kf not in k}
        else:
            results_dict = {k: v for k, v in results_dict.items() if exclude not in k}

    if len(results_dict) == 0:
        print('empty dict')
        return None

    def _file_to_suffix(file_name_str):
        return f"{file_name_str.split('_')[-2]}_{file_name_str.split('_')[-1]}"

    metrics_list = list(results_dict.values())[0][agg].keys().to_list()
    wd_vals = set()
    for k in results_dict.keys():
        wd_vals.add(_file_to_suffix(k))
    colors = list(mcolors.BASE_COLORS.keys())
    c_map = {wdv: colors[i] for i, wdv in enumerate(wd_vals)}

    fig, axs = plt.subplots(2, 2)
    for i, m in enumerate(metrics_list):
        for k, v in results_dict.items():
            axs[i // 2, i % 2].plot(v[agg][m][start_idx:], '--' * ('Asym' in k), c=c_map[_file_to_suffix(k)])
            df_ = v[agg][m][start_idx:]
            x = np.array(df_.index)
            y = np.array(df_)
            y_err = np.array(v['confident'][m][start_idx:])
            axs[i // 2, i % 2].fill_between(x, y - y_err, y + y_err, color=c_map[_file_to_suffix(k)], alpha=0.1)


        if const_val is not None:
            x_range = list(results_dict.values())[0]['mean'].index
            axs[i // 2, i % 2].plot([x_range[start_idx], x_range[-1]], [const_val, const_val], color='k', linestyle='dotted')

        for k, v in sorted(c_map.items()):
            axs[i // 2, i % 2].plot([], [], color=v, label=k)
            # axs[i // 2, i % 2].plot([], [], color=v, label=_file_to_suffix(k))

        axs[i // 2, i % 2].plot([], [], color='k', linestyle='-', label='Symmetric')
        axs[i // 2, i % 2].plot([], [], color='k', linestyle='--', label='Asymmetric')


        axs[i // 2, i % 2].legend()
        axs[i // 2, i % 2].set_title(m)

    plt.show()


def _get_res_dict(directory_path):
    res_dict = {}
    for filename in os.listdir(directory_path):
        f = os.path.join(directory_path, filename)
        # checking if it is a file
        if not (os.path.isfile(f) and 'csv' in filename):
            continue

        results_df = pd.read_csv(f)
        res_dict[filename[:-4]] = {
            'mean': results_df.groupby(['epoch']).agg(np.mean),
            'confident': results_df.groupby(['epoch']).agg(conf_int)
        }

    return res_dict


if __name__ == '__main__':
    path_dir = ''
    res_dict = _get_res_dict(path_dir)

    _generate_comparison_plot(res_dict, key_filter=[], start_idx=0, exclude=[])

    print('done')


# method:
#     metric:
#         mean: [(epoch, score)]
#         std: [(epoch, score)]






