import numpy as np
import torch
from scipy import linalg as scipy_linalg
COVERAGE_K = 5

def hubert_fid(enc_real, enc_gen) -> float:
    real = np.asarray(enc_real, dtype=np.float64)
    gen = np.asarray(enc_gen, dtype=np.float64)
    (mu_r, mu_g) = (real.mean(0), gen.mean(0))
    sig_r = np.cov(real, rowvar=False)
    sig_g = np.cov(gen, rowvar=False)
    diff = mu_r - mu_g
    (covmean, _) = scipy_linalg.sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))
    return max(fid, 0.0)

def real_manifold_radii(enc_real, k=COVERAGE_K, device='cpu'):
    real = torch.from_numpy(np.asarray(enc_real, dtype=np.float32)).to(device)
    distances = torch.cdist(real, real, p=2)
    distances.fill_diagonal_(float('inf'))
    radii = distances.kthvalue(k, dim=1).values
    return radii.cpu().numpy().astype(np.float32, copy=False)
MMD_BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)

def _real_reference_bandwidths(enc_real, max_samples=2048):
    X = torch.from_numpy(np.asarray(enc_real[:max_samples], dtype=np.float32))
    median_distance = float(torch.pdist(X, p=2).median())
    return tuple((median_distance * m for m in MMD_BANDWIDTH_MULTIPLIERS))

def _mmd2_rbf(enc_real, enc_gen, bandwidths) -> float:
    X = torch.from_numpy(np.asarray(enc_real, dtype=np.float32))
    Y = torch.from_numpy(np.asarray(enc_gen, dtype=np.float32))
    sq_xx = torch.cdist(X, X, p=2).square()
    sq_yy = torch.cdist(Y, Y, p=2).square()
    sq_xy = torch.cdist(X, Y, p=2).square()
    mmd2 = 0.0
    for bandwidth in bandwidths:
        kxx = torch.exp(-sq_xx / (2.0 * bandwidth ** 2))
        kyy = torch.exp(-sq_yy / (2.0 * bandwidth ** 2))
        kxy = torch.exp(-sq_xy / (2.0 * bandwidth ** 2))
        (nx, ny) = (len(X), len(Y))
        kxx_mean = (kxx.sum() - kxx.trace()) / (nx * (nx - 1))
        kyy_mean = (kyy.sum() - kyy.trace()) / (ny * (ny - 1))
        mmd2 += float(kxx_mean + kyy_mean - 2.0 * kxy.mean())
    return float(mmd2 / len(bandwidths))

def _density_coverage(enc_real, enc_gen, real_radii, *, k=5, batch_size=256, device='cpu'):
    real = torch.from_numpy(np.asarray(enc_real, dtype=np.float32)).to(device)
    generated = np.asarray(enc_gen, dtype=np.float32)
    radii = torch.from_numpy(np.asarray(real_radii, dtype=np.float32)).to(device)
    density_hits = 0
    nearest_generated = torch.full((len(real),), float('inf'), dtype=real.dtype, device=device)
    for start in range(0, len(generated), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(generated[start:start + batch_size])).to(device)
        distances = torch.cdist(batch, real, p=2)
        density_hits += int((distances <= radii.unsqueeze(0)).sum().item())
        nearest_generated = torch.minimum(nearest_generated, distances.min(dim=0).values)
    density = density_hits / (float(k) * len(generated))
    coverage = float((nearest_generated <= radii).float().mean().item())
    return (float(density), coverage)
