from dataclasses import dataclass

import numpy as np
from utils import z_normalize


@dataclass
class ExperimentConfig:
    n_ts: int = 10
    len_ts: int = 1000
    n_dims: int = 1
    ts_noise_std: float = 0.5

    len_base_motif: int = 100
    m_warping_std: float = 0.0
    len_std: float = 0.0


class ExperimentGeneratorBase:
    def __init__(self, config):
        self.config = config

    def generate_experiment(self):
        ts_list, m_pos_list = [], []

        for i in range(self.config.n_ts):
            ts = self.generate_noisy_ts()
            motif_for_ts = self.motifs[i]
            pos = self.place_motif_randomly(ts, motif_for_ts)
            ts = z_normalize(ts)
            ts_list.append(ts)
            m_pos_list.append(
                (
                    pos + i * self.config.len_ts,
                    pos + motif_for_ts.shape[0] + i * self.config.len_ts,
                )
            )

        return ts_list, m_pos_list

    def generate_base_motif(self):
        length = self.config.len_base_motif
        n_dims = self.config.n_dims

        t = np.linspace(0, 2 * np.pi, length)
        motif = np.zeros((length, n_dims))

        for i in range(n_dims):
            phase = i * (2 * np.pi / n_dims)
            freqs = [1, 2, 3, 5]
            amps = [1, 0.5, 0.3, 0.2]
            for freq, amp in zip(freqs, amps):
                motif[:, i] += amp * np.sin(freq * t + phase)
            motif[:, i] += 0.1 * i * np.linspace(-1, 1, length)

        return motif

    def generate_noisy_ts(self):
        length = self.config.len_ts
        n_dims = self.config.n_dims
        ts = np.zeros((length, n_dims))

        for d in range(n_dims):
            noise = np.random.normal(0, self.config.ts_noise_std, length)
            noise = np.convolve(noise, np.ones(5) / 5, mode='same')
            ts[:, d] = noise

        return ts

    def place_motif_randomly(self, ts, motif):
        length_ts = ts.shape[0]
        length_motif = motif.shape[0]
        pos = np.random.randint(0, length_ts - length_motif + 1)

        ts[pos : pos + length_motif, :] = motif
        return pos


class ExperimentStaticMotif(ExperimentGeneratorBase):
    def __init__(self, config):
        super().__init__(config)
        motif = self.generate_base_motif()
        self.motifs = [motif for _ in range(config.n_ts)]


class ExperimentWarpedMotif(ExperimentGeneratorBase):
    def __init__(self, config):
        super().__init__(config)
        motif = self.generate_base_motif()
        self.motifs = [self.post_process_motif(motif) for _ in range(config.n_ts)]

    def post_process_motif(self, motif):
        return self.add_warping(motif)

    def add_warping(self, motif):
        length = motif.shape[0]
        warping_std = self.config.m_warping_std

        cps = np.random.randint(3, 6)
        cx = np.linspace(0, length - 1, cps)
        cy = cx + np.random.normal(loc=0, scale=warping_std * length, size=cps)

        cy[0] = 0
        cy[-1] = length - 1

        x = np.arange(length)
        warped_x = np.interp(x, cx, cy)

        warped = np.zeros_like(motif)
        for d in range(motif.shape[1]):
            warped[:, d] = np.interp(x, warped_x, motif[:, d])

        return warped


class ExperimentWarpedVariableLengthMotif(ExperimentWarpedMotif):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.motifs = [
            self.post_process_motif(self.generate_base_motif())
            for _ in range(config.n_ts)
        ]

    def generate_base_motif(self):
        mean_len = self.config.len_base_motif
        len_std = self.config.len_std

        length = int(max(5, np.round(np.random.normal(mean_len, len_std))))
        print(length)

        n_dims = self.config.n_dims

        t = np.linspace(0, 2 * np.pi, length)
        motif = np.zeros((length, n_dims))

        for i in range(n_dims):
            phase = i * (2 * np.pi / n_dims)
            freqs = [1, 2, 3, 5]
            amps = [1, 0.5, 0.3, 0.2]
            for freq, amp in zip(freqs, amps):
                motif[:, i] += amp * np.sin(freq * t + phase)
            motif[:, i] += 0.1 * i * np.linspace(-1, 1, length)

        return motif
