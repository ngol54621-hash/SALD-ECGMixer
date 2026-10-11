import numpy as np
from scipy.signal import welch
N_PROJECTIONS = 1024
SEED = 2026

def extract(array):
    (spectral, correlations) = ([], [])
    upper = np.triu_indices(8, 1)
    for start in range(0, len(array), 128):
        x = np.array(array[start:start + 128], dtype=np.float64, copy=True)
        (f, power) = welch(x, fs=100.0, window='hann', nperseg=200, noverlap=100, nfft=200, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        power = power[..., f > 0]
        totals = power.sum(axis=-1, keepdims=True)
        relative = np.divide(power, totals, out=np.zeros_like(power), where=totals > 0)
        spectral.append(np.sqrt(relative).reshape(len(x), -1))
        x -= x.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        centered_unit = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
        corr = centered_unit @ centered_unit.transpose(0, 2, 1)
        correlations.append(np.clip(corr[:, upper[0], upper[1]], -1.0, 1.0))
    return ({'PSD-SWD': np.concatenate(spectral), 'Corr-SWD': np.concatenate(correlations)})

def directions(dimension, stream):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, stream]))
    vectors = rng.standard_normal((N_PROJECTIONS, dimension))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.T

def sorted_projection(features, vectors):
    return np.sort(features @ vectors, axis=0)

def distance(reference_sorted, generated_sorted):
    return float(np.abs(reference_sorted - generated_sorted).mean())
