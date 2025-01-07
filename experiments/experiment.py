import pickle
import time

import locomotif.locomotif as locomotif
import loconsensus.loconsensus as loconsensus
import numpy as np
import stumpy
import tsmd_evaluation.prom as prom
from distancematrix.ostinato import OstinatoAnytime
from experiment_setup import (
    ExperimentConfig,
    ExperimentStaticMotif,
    ExperimentWarpedMotif,
    ExperimentWarpedVariableLengthMotif,
)


###############################################################################
#                               EXPERIMENT 1                                  #
###############################################################################
def run_experiment1(num_runs=100):
    loconsensus_runtimes = []
    locomotif_runtimes = []

    loconsensus_f1_list = []
    locomotif_f1_list = []

    for run in range(1, num_runs + 1):
        config = ExperimentConfig(n_ts=5, len_ts=2000, n_dims=3, len_base_motif=50)
        gen = ExperimentStaticMotif(config)
        ts_list, m_pos = gen.generate_experiment()

        concat_list = np.concatenate(ts_list, axis=0)

        l_min = config.len_base_motif // 2
        l_max = config.len_base_motif * 2
        rho = 0.8
        nb = None

        start_time = time.perf_counter()
        motifs1 = loconsensus.apply_loconsensus(ts_list, l_min, l_max, rho, nb)
        runtime_loconsensus = time.perf_counter() - start_time
        loconsensus_runtimes.append(runtime_loconsensus)

        start_time = time.perf_counter()
        motifs2 = locomotif.apply_locomotif(concat_list, l_min, l_max, rho, nb)
        runtime_locomotif = time.perf_counter() - start_time
        locomotif_runtimes.append(runtime_locomotif)

        M1, _, _ = prom.matching_matrix(
            [m_pos], [motif_set[1] for motif_set in motifs1]
        )
        f1_loconsensus = prom.micro_averaged_f1(M1)
        loconsensus_f1_list.append(f1_loconsensus)

        M2, _, _ = prom.matching_matrix(
            [m_pos], [motif_set[1] for motif_set in motifs2]
        )
        f1_locomotif = prom.micro_averaged_f1(M2)
        locomotif_f1_list.append(f1_locomotif)

        if run % 10 == 0:
            print(f'Experiment 1: Completed run {run}/{num_runs}')

    avg_loconsensus_runtime = np.mean(loconsensus_runtimes)
    avg_locomotif_runtime = np.mean(locomotif_runtimes)

    loconsensus_perfect = sum(1 for f in loconsensus_f1_list if abs(f - 1.0) < 1e-9)
    locomotif_perfect = sum(1 for f in locomotif_f1_list if abs(f - 1.0) < 1e-9)

    results = {
        'experiment': 'Experiment 1',
        'num_runs': num_runs,
        'loconsensus': {
            'average_runtime_sec': avg_loconsensus_runtime,
            'f1_scores': loconsensus_f1_list,
            'perfect_f1_runs': loconsensus_perfect,
        },
        'locomotif': {
            'average_runtime_sec': avg_locomotif_runtime,
            'f1_scores': locomotif_f1_list,
            'perfect_f1_runs': locomotif_perfect,
        },
    }

    print('===== EXPERIMENT 1 RESULTS =====')
    print(f'Number of runs: {num_runs}')
    print(f'LoConsensus average runtime : {avg_loconsensus_runtime:.4f} s')
    print(f'LoCoMotif   average runtime : {avg_locomotif_runtime:.4f} s')
    print(f'LoConsensus perfect-F1 runs : {loconsensus_perfect} / {num_runs}')
    print(f'LoCoMotif   perfect-F1 runs : {locomotif_perfect} / {num_runs}')
    print()

    return results


###############################################################################
#                               EXPERIMENT 2                                  #
###############################################################################
def run_experiment2(exp, runs=50):
    loconsensus_runtimes = []
    ostinato_runtimes = []
    anytime_ostinato_runtimes = []

    loconsensus_f1_list = []
    ostinato_f1_list = []
    anytime_ostinato_f1_list = []

    def top1(radii, flat_list):
        min_dist_per_series = [np.min(radii[i]) for i in range(len(flat_list))]
        min_idx_per_series = [np.argmin(radii[i]) for i in range(len(flat_list))]
        series_idx = np.argmin(min_dist_per_series)
        subseq_idx = min_idx_per_series[series_idx]
        return series_idx, subseq_idx

    for run in range(1, runs + 1):
        if exp == 1:
            config = ExperimentConfig()
            gen = ExperimentStaticMotif(config)
        elif exp == 2:
            config = ExperimentConfig(m_warping_std=0.5, len_std=0.0)
            gen = ExperimentWarpedMotif(config)
        elif exp == 3:
            config = ExperimentConfig(m_warping_std=0.25, len_std=25)
            gen = ExperimentWarpedVariableLengthMotif(config)
        else:
            raise ValueError(f'Invalid exp={exp}. Choose 1, 2, or 3.')

        ts_list, m_pos = gen.generate_experiment()
        flat_list = [ts.flatten() for ts in ts_list]

        len_base_motif = config.len_base_motif
        len_ts = config.len_ts

        l_min = len_base_motif // 2
        l_max = len_base_motif * 2
        rho = 0.8
        nb = None

        start_time = time.perf_counter()
        motifs1 = loconsensus.apply_loconsensus(ts_list, l_min, l_max, rho, nb)
        runtime_loconsensus = time.perf_counter() - start_time
        loconsensus_runtimes.append(runtime_loconsensus)

        M1, _, _ = prom.matching_matrix(
            [m_pos], [motif_set[1] for motif_set in motifs1]
        )
        f1_loconsensus = prom.micro_averaged_f1(M1)
        loconsensus_f1_list.append(f1_loconsensus)

        start_time = time.perf_counter()
        r, series_idx, subseq_idx = stumpy.ostinato(flat_list, len_base_motif)

        motif_candidate = flat_list[series_idx][
            subseq_idx : subseq_idx + len_base_motif
        ]
        nn1 = []
        for i, ts in enumerate(flat_list):
            dist_profile = stumpy.core.mass(motif_candidate, ts)
            ms = np.argmin(dist_profile)
            nn1.append((ms + i * len_ts, ms + len_base_motif + i * len_ts))
        runtime_ostinato = time.perf_counter() - start_time
        ostinato_runtimes.append(runtime_ostinato)

        M2, _, _ = prom.matching_matrix([m_pos], [nn1])
        f1_ostinato = prom.micro_averaged_f1(M2)
        ostinato_f1_list.append(f1_ostinato)

        start_time = time.perf_counter()
        oa = OstinatoAnytime(flat_list, len_base_motif)
        oa.calculate(1.0)
        r1 = oa.get_radii()
        runtime_anytime_ostinato = time.perf_counter() - start_time
        anytime_ostinato_runtimes.append(runtime_anytime_ostinato)

        nn2 = []
        for i in range(len(flat_list)):
            ms = np.argmin(r1[i])
            nn2.append((ms + i * len_ts, ms + len_base_motif + i * len_ts))

        M3, _, _ = prom.matching_matrix([m_pos], [nn2])
        f1_anytime_ostinato = prom.micro_averaged_f1(M3)
        anytime_ostinato_f1_list.append(f1_anytime_ostinato)

        if run % 10 == 0:
            print(f'Experiment 2 (exp={exp}): Completed run {run}/{runs}')

    avg_loconsensus_runtime = np.mean(loconsensus_runtimes)
    avg_ostinato_runtime = np.mean(ostinato_runtimes)
    avg_anytime_ostinato_runtime = np.mean(anytime_ostinato_runtimes)

    avg_loconsensus_f1 = np.mean(loconsensus_f1_list)
    avg_ostinato_f1 = np.mean(ostinato_f1_list)
    avg_anytime_ostinato_f1 = np.mean(anytime_ostinato_f1_list)

    experiment_name = f'Experiment 2 (exp={exp})'
    results = {
        'experiment': experiment_name,
        'num_runs': runs,
        'loconsensus': {
            'average_runtime_sec': avg_loconsensus_runtime,
            'f1_scores': loconsensus_f1_list,
            'average_f1': avg_loconsensus_f1,
        },
        'ostinato': {
            'average_runtime_sec': avg_ostinato_runtime,
            'f1_scores': ostinato_f1_list,
            'average_f1': avg_ostinato_f1,
        },
        'anytime_ostinato': {
            'average_runtime_sec': avg_anytime_ostinato_runtime,
            'f1_scores': anytime_ostinato_f1_list,
            'average_f1': avg_anytime_ostinato_f1,
        },
    }

    print(f'===== {experiment_name} RESULTS =====')
    print(f'Number of runs: {runs}')
    print(
        f'[LoConsensus]       Avg runtime: {avg_loconsensus_runtime:.4f} s | Avg F1: {avg_loconsensus_f1:.4f}'
    )
    print(
        f'[Stumpy Ostinato]   Avg runtime: {avg_ostinato_runtime:.4f} s   | Avg F1: {avg_ostinato_f1:.4f}'
    )
    print(
        f'[Anytime Ostinato]  Avg runtime: {avg_anytime_ostinato_runtime:.4f} s | Avg F1: {avg_anytime_ostinato_f1:.4f}'
    )
    print()

    return results


def save_results_to_pickle(results, filename='experiment_results.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    experiment1_results = run_experiment1(num_runs=100)

    experiment2_results_exp1 = run_experiment2(exp=1, runs=50)
    experiment2_results_exp3 = run_experiment2(exp=3, runs=50)

    all_results = {
        'experiment1': experiment1_results,
        'experiment2_exp1': experiment2_results_exp1,
        'experiment2_exp3': experiment2_results_exp3,
    }

    save_results_to_pickle(all_results, filename='experiment_results.pkl')
