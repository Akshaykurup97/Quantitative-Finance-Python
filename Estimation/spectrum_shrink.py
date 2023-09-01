
import numpy as np
from scipy.interpolate import interp1d
from arpym.statistics.marchenko_pastur import marchenko_pastur
from arpym.tools.pca_cov import pca_cov


def spectrum_shrink(sigma2_in, t_):
   

    i_ = sigma2_in.shape[0]

    # PCA decomposition

    e, lam = pca_cov(sigma2_in)

    # Determine optimal k_
    ll = 1000

    dist = np.ones(i_-1)*np.nan

    for k in range(i_-1):

        lam_k = lam[k+1:]
        lam_noise = np.mean(lam_k)

        q = t_/len(lam_k)

        # compute M-P on a very dense grid
        x_tmp, mp_tmp = marchenko_pastur(lam_noise, q, ll)
        
        if q > 1:
            x_tmp = np.r_[0, x_tmp[0], x_tmp]
            mp_tmp = np.r_[0, mp_tmp[0], mp_tmp]
        l_max = np.max(lam_k)
        if l_max > x_tmp[-1]:
            x_tmp = np.r_[x_tmp, x_tmp[-1], l_max]
            mp_tmp = np.r_[mp_tmp, 0, 0]

        # compute the histogram of eigenvalues
        hgram, x_bin_edge = np.histogram(lam_k, bins='auto', density=True)
        bin_size = np.diff(x_bin_edge)[0]
        x_bin = x_bin_edge[:-1]+bin_size/2
        
        # interpolation
        interp = interp1d(x_tmp, mp_tmp, fill_value='extrapolate')
        mp = interp(x_bin)

        dist[k] = np.mean((mp-hgram)**2)

    err_tmp, k_tmp = np.nanmin(dist), np.nanargmin(dist)
    k_ = k_tmp
    err = err_tmp

    # Isotropy
    lam_out = lam
    lam_noise = np.mean(lam[k_+1:])

    lam_out[k_+1:] = lam_noise  # shrunk spectrum

    # Output

    sigma2_out = e@np.diagflat(lam_out)@e.T

    # compute M-P on a very dense grid
    x_mp, y_mp = marchenko_pastur(lam_noise, t_/(i_-k_-1), 100)

    return sigma2_out, lam_out, k_, err, y_mp, x_mp, dist


