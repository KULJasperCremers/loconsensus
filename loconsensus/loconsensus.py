import multiprocessing
from itertools import combinations_with_replacement

import numpy as np
from joblib import Parallel, delayed
from numba import boolean, float32, float64, int32, njit, prange, typed, types
from numba.experimental import jitclass

from loconsensus.utils import find_timeseries_index, offset_indexer, row_col_from_cindex


def apply_loconsensus(ts_list, l_min, l_max, rho, nb=None, overlap=0.0):
    """Apply the LoCosensus algorithm to find consensus motifs in a list of timeseries.

    Args:
        ts_list: list of univariate or multivariate timeseries, shape: (datapoints, dim)
        l_min: minimum candidate motif length
        l_max: maximum candidate motif length
        nb: maximum number of (consensus) motif tuples to find
        overlap: maximum allowed overlap between motifs, [0,0.5]

    Returns: cmotif_sets: a list of (consensus) motif tuples

    """
    ts_lengths = [len(ts) for ts in ts_list]

    n = len(ts_list)
    offset_indices = offset_indexer(n)
    global_offsets = np.cumsum([0] + ts_lengths, dtype=np.int32)
    # total_comparisons = n * (n + 1) // 2

    lccs = []
    args_list = []

    for cindex, (ts1, ts2) in enumerate(combinations_with_replacement(ts_list, 2)):
        lcc = get_lococonsensus_instance(
            ts1, ts2, global_offsets, offset_indices[cindex], l_min, rho, cindex, n
        )
        lccs.append(lcc)
        args_list.append(lcc)

    n_threads = multiprocessing.cpu_count()

    def process_comparison(lcc):
        lcc.apply_loco()

    Parallel(n_jobs=n_threads, backend='threading')(
        delayed(process_comparison)(args) for args in args_list
    )

    mc = get_motifconsensus_instance(n, global_offsets, l_min, l_max, lccs)

    motif_sets = []
    for motif in mc.apply_motif(nb, overlap):
        motif_sets.append(motif)

    return motif_sets


def get_lococonsensus_instance(
    ts1, ts2, global_offsets, offset_indices, l_min, rho, cindex, n
):
    is_diagonal = False
    r, c = row_col_from_cindex(cindex, n)
    if r == c:
        is_diagonal = True
    ts1 = np.array(ts1, dtype=np.float32)
    ts2 = np.array(ts2, dtype=np.float32)

    gamma = 1
    sm, ut_sm = None, None
    if is_diagonal:
        sm = calculate_similarity_matrix(ts1, ts2, gamma, only_triu=is_diagonal)
        tau = estimate_tau_symmetric(sm, rho)
    else:
        ut_sm = calculate_similarity_matrix(ts1, ts2, gamma, only_triu=is_diagonal)
        tau = estimate_tau_assymmetric(ut_sm, rho)

    delta_a = 2 * tau
    delta_m = 0.5
    step_sizes = np.array([(1, 1), (2, 1), (1, 2)])
    lcs = LoCoConsensus(
        ts1=ts1,
        ts2=ts2,
        is_diagonal=is_diagonal,
        l_min=l_min,
        gamma=gamma,
        tau=tau,
        delta_a=delta_a,
        delta_m=delta_m,
        step_sizes=step_sizes,
        global_offsets=global_offsets,
        offset_indices=offset_indices,
    )
    lcs._sm = (sm, ut_sm)
    return lcs


class LoCoConsensus:
    def __init__(
        self,
        ts1,
        ts2,
        is_diagonal,
        l_min,
        gamma,
        tau,
        delta_a,
        delta_m,
        step_sizes,
        global_offsets,
        offset_indices,
    ):
        self.ts1 = ts1
        self.is_diagonal = is_diagonal
        self.ts2 = ts2
        self.l_min = np.int32(l_min)
        self.step_sizes = step_sizes.astype(np.int32)

        self.gamma = gamma
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m

        self._sm = (None, None)
        self._csm = None
        self._paths = None
        self._mirrored_paths = None

        self.rstart = global_offsets[offset_indices[0]]
        self.cstart = global_offsets[offset_indices[1]]

    def apply_loco(self):
        self._align()
        self._find_best_paths(vwidth=self.l_min // 2)

    def _align(self):
        if self.is_diagonal:
            sm = self._sm[0]
        else:
            sm = self._sm[1]
        self._csm = calculate_cumulative_similarity_matrix(
            sm,
            tau=self.tau,
            delta_a=self.delta_a,
            delta_m=self.delta_m,
            step_sizes=self.step_sizes,
            only_triu=self.is_diagonal,
        )

    def _find_best_paths(self, vwidth):
        if self.is_diagonal:
            mask = np.full(self._csm.shape, True)
            mask[np.triu_indices(len(mask), k=vwidth)] = False
            diagonal = np.vstack(np.diag_indices(len(self.ts1))).T
            gdiagonal = diagonal + [self.rstart, self.cstart]
        else:
            mask = np.full(self._csm.shape, False)
            self._mirrored_paths = typed.List()
        found_paths = _find_best_paths(
            self._csm, mask, l_min=self.l_min, vwidth=vwidth, step_sizes=self.step_sizes
        )

        self._paths = typed.List()

        if self.is_diagonal:
            self._paths.append(
                GlobalPath(
                    gdiagonal.astype(np.int32),
                    np.ones(len(diagonal)).astype(np.float32),
                )
            )

        for path in found_paths:
            i, j = path[:, 0], path[:, 1]
            gpath = np.zeros(path.shape, dtype=np.int32)
            gpath[:, 0] = np.copy(path[:, 0]) + self.rstart
            gpath[:, 1] = np.copy(path[:, 1]) + self.cstart
            gpath_mir = np.zeros(path.shape, dtype=np.int32)

            if self.is_diagonal:
                path_sims = self._sm[0][i, j]
                self._paths.append(GlobalPath(gpath, path_sims))
                gpath_mir[:, 0] = np.copy(path[:, 1]) + self.rstart
                gpath_mir[:, 1] = np.copy(path[:, 0]) + self.rstart
                self._paths.append(GlobalPath(gpath_mir, path_sims))
            else:
                path_sims = self._sm[1][i, j]
                self._paths.append(GlobalPath(gpath, path_sims))
                mir_path_sims = self._sm[1].T[j, i]
                gpath_mir[:, 0] = np.copy(path[:, 1]) + self.cstart
                gpath_mir[:, 1] = np.copy(path[:, 0]) + self.rstart
                self._mirrored_paths.append(GlobalPath(gpath_mir, mir_path_sims))

    def get_paths(self):
        if self._mirrored_paths is not None:
            paths = [path.path for path in self._paths]
            paths += [path.path for path in self._mirrored_paths]
            return paths
        else:
            return [path.path for path in self._paths]

    def get_sm(self):
        sm = self._sm[0] if self.is_diagonal else self._sm[1]
        return sm


@njit(float32[:, :](float32[:, :], float32[:, :], int32, boolean))
def calculate_similarity_matrix(ts1, ts2, gamma, only_triu):
    n, m = len(ts1), len(ts2)
    similarity_matrix = np.full((n, m), -np.inf, dtype=np.float32)
    for i in range(n):
        j_start = i if only_triu else 0
        j_end = m
        similarities = np.exp(
            -gamma * np.sum(np.power(ts1[i, :] - ts2[j_start:j_end, :], 2), axis=1)
        )
        similarity_matrix[i, j_start:j_end] = similarities
    return similarity_matrix


@njit(float32[:, :](float32[:, :], float64, float64, float64, int32[:, :], boolean))
def calculate_cumulative_similarity_matrix(
    sm, tau, delta_a, delta_m, step_sizes, only_triu
):
    n, m = sm.shape
    max_v = np.amax(step_sizes[:, 0])
    max_h = np.amax(step_sizes[:, 1])

    csm = np.zeros((n + max_v, m + max_h), dtype=np.float32)
    for i in range(n):
        j_start = i if only_triu else 0
        j_end = m
        for j in range(j_start, j_end):
            sim = sm[i, j]

            indices = np.array([i + max_v, j + max_h]) - step_sizes
            max_cumsim = np.amax(np.array([csm[i_, j_] for (i_, j_) in indices]))

            if sim < tau:
                csm[i + max_v, j + max_h] = max(0, delta_m * max_cumsim - delta_a)
            else:
                csm[i + max_v, j + max_h] = max(0, sim + max_cumsim)
    return csm


@njit(int32[:, :](float32[:, :], boolean[:, :], int32, int32))
def max_warping_path(csm, mask, i, j):
    # tie-breaker
    r, c = csm.shape
    if r >= c:
        step_sizes = np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)
    else:
        step_sizes = np.array([[1, 1], [1, 2], [2, 1]], dtype=np.int32)

    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    path = []
    while i >= max_v and j >= max_h:
        path.append((i - max_v, j - max_h))
        indices = np.array([i, j], dtype=np.int32) - step_sizes
        values = np.array([csm[i_, j_] for (i_, j_) in indices])
        masked = np.array([mask[i_, j_] for (i_, j_) in indices])
        argmax = np.argmax(values)

        if masked[argmax]:
            break

        i, j = i - step_sizes[argmax, 0], j - step_sizes[argmax, 1]

    path.reverse()
    return np.array(path, dtype=np.int32)


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32))
def mask_path(path, mask, v, h):
    for x, y in path:
        mask[x + h, y + v] = True
    return mask


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32, int32))
def mask_vicinity(path, mask, v, h, vwidth):
    (xc, yc) = path[0] + np.array((v, h))
    for xt, yt in path[1:] + np.array([v, h]):
        dx = xt - xc
        dy = yc - yt
        err = dx + dy
        while xc != xt or yc != yt:
            mask[xc - vwidth : xc + vwidth + 1, yc] = True
            mask[xc, yc - vwidth : yc + vwidth + 1] = True
            e = 2 * err
            if e > dy:
                err += dy
                xc += 1
            if e < dx:
                err += dx
                yc += 1
    mask[xt - vwidth : xt + vwidth + 1, yt] = True
    mask[xt, yt - vwidth : yt + vwidth + 1] = True
    return mask


@njit(types.List(int32[:, :])(float32[:, :], boolean[:, :], int32, int32, int32[:, :]))
def _find_best_paths(csm, mask, l_min, vwidth, step_sizes):
    max_v = max(step_sizes[:, 0])
    max_h = max(step_sizes[:, 1])

    is_, js_ = np.nonzero(csm <= 0)
    for index_best in range(len(is_)):
        mask[is_[index_best], js_[index_best]] = True

    is_, js_ = np.nonzero(csm)
    values = np.array([csm[is_[i], js_[i]] for i in range(len(is_))])
    perm = np.argsort(values)
    is_ = is_[perm]
    js_ = js_[perm]

    index_best = len(is_) - 1
    paths = []

    while index_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False
        while not path_found:
            while mask[is_[index_best], js_[index_best]]:
                index_best -= 1
                if index_best < 0:
                    return paths

            i_best, j_best = is_[index_best], js_[index_best]

            if i_best < max_v or j_best < max_h:
                return paths

            path = max_warping_path(csm, mask, i_best, j_best)
            mask = mask_path(path, mask, max_v, max_h)

            if (path[-1][0] - path[0][0] + 1) >= l_min or (
                path[-1][1] - path[0][1] + 1
            ) >= l_min:
                path_found = True

        mask = mask_vicinity(path, mask, max_v, max_h, vwidth)
        paths.append(path)

    return paths


def estimate_tau_symmetric(sm, rho):
    tau = np.quantile(sm[np.triu_indices(len(sm))], rho, axis=None)
    return tau


def estimate_tau_assymmetric(sm, rho):
    tau = np.quantile(sm, rho, axis=None)
    return tau


spec = [
    ('path', int32[:, :]),
    ('sim', float32[:]),
    ('cumsim', float32[:]),
    ('index_gi', int32[:]),
    ('index_gj', int32[:]),
    ('gi1', int32),
    ('gil', int32),
    ('gj1', int32),
    ('gjl', int32),
]


@jitclass(spec)
class GlobalPath:
    def __init__(self, path, sim):
        assert len(path) == len(sim)
        self.path = path
        self.sim = sim.astype(np.float32)
        self.cumsim = np.concatenate(
            (np.array([0.0], dtype=np.float32), np.cumsum(sim))
        )
        self.gi1 = path[0][0]
        self.gil = path[len(path) - 1][0] + 1
        self.gj1 = path[0][1]
        self.gjl = path[len(path) - 1][1] + 1
        self._construct_global_index(path)

    def __getitem__(self, i):
        return self.path[i, :]

    def __len__(self):
        return len(self.path)

    def _construct_global_index(self, path):
        i_curr = path[0][0]
        j_curr = path[0][1]

        index_gi = np.zeros(self.gil - self.gi1, dtype=np.int32)
        index_gj = np.zeros(self.gjl - self.gj1, dtype=np.int32)

        for i in range(1, len(path)):
            if path[i][0] != i_curr:
                index_gi[i_curr - self.gi1 + 1 : path[i][0] - self.gi1 + 1] = i
                i_curr = path[i][0]

            if path[i][1] != j_curr:
                index_gj[j_curr - self.gj1 + 1 : path[i][1] - self.gj1 + 1] = i
                j_curr = path[i][1]

        self.index_gi = index_gi
        self.index_gj = index_gj

    def find_gi(self, i):
        assert i - self.gi1 >= 0 and i - self.gi1 < len(self.index_gi)
        return self.index_gi[i - self.gi1]

    def find_gj(self, j):
        assert j - self.gj1 >= 0 and j - self.gj1 < len(self.index_gj)
        return self.index_gj[j - self.gj1]


def get_motifconsensus_instance(n, global_offsets, l_min, l_max, lccs):
    gcs = []
    for column_index in range(n):
        gcs.append(GlobalColumn(column_index, global_offsets, l_min, l_max))

    i = 0
    for r in range(n):
        for c in range(r, n):
            lcc = lccs[i]
            i += 1

            if r == c:
                gcs[r].append_paths(lcc._paths)
            else:
                gcs[c].append_paths(lcc._paths)
                if lcc._mirrored_paths:
                    gcs[r].append_paths(lcc._mirrored_paths)

    return MotifConsensus(global_offsets, gcs)


class MotifConsensus:
    def __init__(self, global_offsets, global_columns):
        self.global_offsets = global_offsets
        self.global_columns = global_columns
        self.ccs = [None] * len(global_columns)

    def apply_motif(self, nb, overlap):
        smask = np.full(self.global_offsets[-1], True)
        emask = np.full(self.global_offsets[-1], True)
        mask = np.full(self.global_offsets[-1], False)

        num_threads = multiprocessing.cpu_count()
        current_nb = 0
        while nb is None or current_nb < nb:
            if np.all(mask) and not np.any(smask) or not np.any(emask):
                break

            smask[mask] = False
            emask[mask] = False

            best_fitness = 0.0
            best_candidate = None
            best_cindex = None

            args_list = []
            for cindex, gc in enumerate(self.global_columns):
                if not self.ccs[cindex]:
                    args = (cindex, gc, smask, emask, mask, overlap, False)
                    args_list.append(args)

            results = Parallel(n_jobs=num_threads, backend='threading')(
                delayed(_process_candidate)(args) for args in args_list
            )

            for cindex, candidate, fitness, _ in results:
                if fitness > 0.0:
                    self.ccs[cindex] = (candidate, fitness, _)
                else:
                    self.ccs[cindex] = None

            for cindex, cc in enumerate(self.ccs):
                if not cc:
                    continue
                candidate, fitness, _ = cc
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate
                    best_cindex = cindex

            if best_fitness == 0.0:
                break

            b, e = best_candidate
            gc = self.global_columns[best_cindex]
            ips = gc.induced_paths(b, e, mask)
            motif_set = vertical_projections(ips)
            for bm, em in motif_set:
                l = em - bm
                mask[bm + int(overlap * l) - 1 : em - int(overlap * l)] = True

            """
            This enables the return of local motifs instead. DONT forget to change the yield!

            local_motif_set = []
            for ip in ips:
                lp = vertical_projection(ip, self.global_offsets)
                local_motif_set.append(lp)
            """
            for cindex, cc in enumerate(self.ccs):
                if cindex == best_cindex or not cc:
                    continue
                (b2, e2), _, _ = cc
                gc2 = self.global_columns[cindex]
                ips2 = gc2.induced_paths(b2, e2, mask)
                if np.any(mask[b2:e2]) or len(ips2) < 2:
                    self.ccs[cindex] = None
            self.ccs[best_cindex] = None

            current_nb += 1
            yield (b, e), motif_set, _, best_fitness
            # yield (b, e), local_motif_set, ips, best_fitness


def _process_candidate(args):
    (cindex, gc, smask, emask, mask, overlap, keep_fitnesses) = args
    candidate, best_fitness, _ = gc.candidate_finder(
        smask, emask, mask, overlap, keep_fitnesses
    )

    return cindex, candidate, best_fitness, _


def vertical_projections(paths):
    return [(p[0][0], p[len(p) - 1][0] + 1) for p in paths]


def vertical_projection(path, goffsets):
    idx = find_timeseries_index(path[0][0], goffsets)
    lp = path - (goffsets[idx], goffsets[idx])
    return (lp[0][0], lp[len(lp) - 1][0] + 1)


class GlobalColumn:
    def __init__(self, cindex, global_offsets, l_min, l_max):
        self.global_offsets = global_offsets
        self.global_n = global_offsets[-1]
        self.l_min = l_min
        self.l_max = l_max
        self._column_paths = None

        self.start_offset = global_offsets[cindex]
        self.end_offset = global_offsets[cindex + 1] - 1

    def candidate_finder(self, smask, emask, mask, overlap, keep_fitnesses):
        (b, e), best_fitness, fitnesses = _find_best_candidate(
            smask,
            emask,
            mask,
            self._column_paths,
            self.l_min,
            self.l_max,
            self.start_offset.astype(np.int32),
            self.end_offset.astype(np.int32),
            overlap,
            keep_fitnesses,
        )
        return (b, e), best_fitness, fitnesses

    def append_paths(self, paths):
        if self._column_paths is None:
            self._column_paths = typed.List()
        for path in paths:
            self._column_paths.append(path)

    def append_mpaths(self, mpaths):
        if self._column_paths is None:
            self._column_paths = typed.List()
        for mpath in mpaths:
            self._column_paths.append(mpath)

    def induced_paths(self, b, e, mask):
        induced_paths = []

        for p in self._column_paths:
            if p.gj1 <= b and e <= p.gjl:
                kb, ke = p.find_gj(b), p.find_gj(e - 1)
                bm, em = p[kb][0], p[ke][0] + 1
                if not np.any(mask[bm:em]):
                    ip = np.copy(p.path[kb : ke + 1])
                    induced_paths.append(ip)
        return induced_paths


@njit(
    types.Tuple((types.UniTuple(int32, 2), float32, float32[:, :]))(
        boolean[:],
        boolean[:],
        boolean[:],
        types.ListType(GlobalPath.class_type.instance_type),  # type:ignore
        int32,
        int32,
        int32,
        int32,
        float64,
        boolean,
    )
)
def _find_best_candidate(
    start_mask,
    end_mask,
    mask,
    paths,
    l_min,
    l_max,
    start_offset,
    end_offset,
    overlap=0.5,
    keep_fitnesses=False,
):
    fitnesses = []
    n = len(mask)
    mn = end_offset - start_offset

    j1s = np.array([path.gj1 for path in paths])
    jls = np.array([path.gjl for path in paths])

    nbp = len(paths)

    bs = np.zeros(nbp, dtype=np.int32)
    es = np.zeros(nbp, dtype=np.int32)

    kbs = np.zeros(nbp, dtype=np.int32)
    kes = np.zeros(nbp, dtype=np.int32)

    best_fitness = 0.0
    best_candidate = (0, n)

    for b in prange(mn - l_min + 1):
        gb = b + start_offset
        if not start_mask[gb]:
            continue
        smask = j1s <= gb

        for e in range(b + l_min, min(mn + 1, b + l_max + 1)):
            ge = e + start_offset
            if not end_mask[ge - 1]:
                continue

            if np.any(mask[gb:ge]):
                break

            emask = jls >= ge
            pmask = smask & emask

            # If the candidate only matches with itself, skip it.
            if not np.sum(pmask) > 1:
                break

            for p in np.flatnonzero(pmask):
                path = paths[p]
                kbs[p] = pi = path.find_gj(gb)
                kes[p] = pj = path.find_gj(ge - 1)
                bs[p] = path[pi][0]
                es[p] = path[pj][0] + 1
                if np.any(mask[bs[p] : es[p]]):
                    pmask[p] = False

            # If the candidate only matches with itself, skip it.
            if not np.sum(pmask) > 1:
                break

            bs_ = bs[pmask]
            es_ = es[pmask]

            perm = np.argsort(bs_)
            bs_ = bs_[perm]
            es_ = es_[perm]

            len_ = es_ - bs_
            len_[:-1] = np.minimum(len_[:-1], len_[1:])
            overlaps = np.maximum(es_[:-1] - bs_[1:], 0)

            if np.any(overlaps > overlap * len_[:-1]):
                continue

            coverage = np.sum(es_ - bs_) - np.sum(overlaps)
            n_coverage = coverage / float(n)

            score = 0
            for p in np.flatnonzero(pmask):
                score += paths[p].cumsim[kes[p] + 1] - paths[p].cumsim[kbs[p]]
            n_score = score / float(np.sum(kes[pmask] - kbs[pmask] + 1))

            # weighted harmonic score
            fit = 0.0
            if n_coverage != 0 or n_score != 0:
                # w1, w2 = 0.25, 0.75
                # lcm weights:
                w1, w2 = 0.5, 0.5
                fit = (
                    (w1 + w2)
                    * (n_coverage * n_score)
                    / (w1 * n_coverage + w2 * n_score)
                )

            if fit == 0.0:
                continue

            if fit > best_fitness:
                best_candidate = (gb, ge)
                best_fitness = fit

            if keep_fitnesses:
                fitnesses.append((gb, ge, fit, n_coverage, n_score))

    fitnesses = (
        np.array(fitnesses, dtype=np.float32)
        if fitnesses
        else np.empty((0, 5), dtype=np.float32)
    )
    return best_candidate, best_fitness, fitnesses
