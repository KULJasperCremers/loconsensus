import pickle
from pathlib import Path

import locomotif.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

motifs_file = Path('./data/motifs.pkl')
with motifs_file.open('rb') as f:
    motifs = pickle.load(f)

goffsets_file = Path('./data/goffsets.pkl')
with goffsets_file.open('rb') as f:
    goffsets = pickle.load(f)

series_file = Path('./data/series.pkl')
with series_file.open('rb') as f:
    series = pickle.load(f)

mvis = True
if mvis:
    fig, axs = visualize.plot_motif_sets(series, motifs)
    plt.savefig('./plots/motifs.png')


def find_timeseries_index(gindex, goffsets):
    for i in range(len(goffsets) - 1):
        if goffsets[i] <= gindex < goffsets[i + 1]:
            return i


n = len(goffsets) - 1
sm = np.zeros((n, n))

# yield (b, e), motif_set, csums, _
for mt in motifs:
    cs, ce = mt[0]
    cindex = find_timeseries_index(cs, goffsets)
    for i, motif in enumerate(mt[1]):
        ms, me = motif
        if cs == ms and ce == me:
            continue
        csim = mt[2][i]
        mindex = find_timeseries_index(ms, goffsets)
        sm[mindex][cindex] += csim

smin = sm.min()
smax = sm.max()
nm = (sm - smin) / (smax - smin)

d = 1 - nm
Z = linkage(d, method='average')

labels = [
    'w1',
    'r1',
    'c1',
    'w2',
    'r2',
    'c2',
    'w3',
    #'w4',
    #'c4',
    #'w5',
    #'r5',
    #'c5',
    #'w6',
    #'r6',
    #'c6',
    #'w7',
    #'r7',
    #'c7',
    #'w8',
    #'r8',
    #'c8',
]
plt.figure(figsize=(25, len(labels) * 3))
dendrogram(Z, labels=labels)
plt.savefig('./plots/dendogram.png')
plt.close()
