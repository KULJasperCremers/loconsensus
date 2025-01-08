import pickle
import random
import time
from pathlib import Path

import locomotif.locomotif as locomotif
import locomotif.visualize as visualize
import loconsensus.loconsensus as loconsensus
import matplotlib.pyplot as plt
import numpy as np
import tsmd_evaluation.prom as prom


def create_random_activities_ts(
    subjects, min_length=1000, n_activities=5, max_retries=5
):
    subject_ids = list(subjects.keys())
    retries = 0

    while retries < max_retries:
        random_subject = random.choice(subject_ids)
        all_activities = list(subjects[random_subject].keys())

        valid_activities = [
            activity
            for activity in all_activities
            if subjects[random_subject][activity].shape[0] >= min_length
        ]

        if len(valid_activities) >= n_activities:
            chosen_activities = random.sample(valid_activities, n_activities)
            ts_list = [
                subjects[random_subject][activity][:min_length, 6:9]
                for activity in chosen_activities
            ]
            combined_ts = np.concatenate(ts_list, axis=0)
            return ts_list, combined_ts

        elif len(valid_activities) > 0:
            chosen_activities = random.sample(valid_activities, len(valid_activities))
            ts_list = [
                subjects[random_subject][activity][:min_length, 6:9]
                for activity in chosen_activities
            ]
            combined_ts = np.concatenate(ts_list, axis=0)
            return ts_list, combined_ts

        else:
            retries += 1


s_dir = Path('./pickles/subjects.pkl')
with open(s_dir, 'rb') as f:
    subjects = pickle.load(f)

l_min = 15
l_max = 30
rho = 0.8
nb = None

total_runs = 500
lcc_runtimes = []
lcm_runtimes = []
f1_scores = []

for i in range(total_runs):
    ts_list, combined_ts = create_random_activities_ts(subjects)
    if i == 0:
        for j, ts in enumerate(ts_list):
            fig, ax = visualize.plot_motif_sets(ts, [], legend=False)
            ax[0].set_ylim([-3, 3])
            plt.savefig(f'./plots/ts_{j}.png')
            plt.close()

        fig, ax = visualize.plot_motif_sets(combined_ts, [], legend=False)
        ax[0].set_ylim([-3, 3])
        plt.savefig('./plots/combined_ts.png')
        plt.close()

    s1 = time.perf_counter()
    m1 = loconsensus.apply_loconsensus(ts_list, l_min, l_max, rho, nb)
    e1 = time.perf_counter() - s1
    lcc_runtimes.append(e1)

    s2 = time.perf_counter()
    m2 = locomotif.apply_locomotif(combined_ts, l_min, l_max, rho, nb)
    e2 = time.perf_counter() - s2
    lcm_runtimes.append(e2)

    M, _, _ = prom.matching_matrix([ms[1] for ms in m1], [ms[1] for ms in m2])
    f1 = prom.micro_averaged_f1(M)
    f1_scores.append(f1)

    if i % 100 == 0:
        ms = []
        for _, m, *_ in m1[:5]:
            ms.append(m)

        fig, ax = visualize.plot_motif_sets(combined_ts, ms, legend=False)
        ax[0].set_ylim([-3, 3])
        plt.savefig(f'./plots/lcc_ms_{i}.png')
        plt.close()

        fig, ax = visualize.plot_motif_sets(combined_ts, m2[:5], legend=False)
        ax[0].set_ylim([-3, 3])
        plt.savefig(f'./plots/lcm_ms_{i}.png')
        plt.close()

    if (i + 1) % 25 == 0:
        print(f'Completed {i+1}/{total_runs} runs.')

runtime_data = {
    'loconsensus_runtimes': lcc_runtimes,
    'locomotif_runtimes': lcm_runtimes,
    'f1_scores': f1_scores,
}

runtime_pickle_path = Path('./pickles') / 'runtimes.pkl'
with open(runtime_pickle_path, 'wb') as f:
    pickle.dump(runtime_data, f)
