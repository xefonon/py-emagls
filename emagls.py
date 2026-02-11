import numpy as np
from utils import (
    cart2sph,
    sph_harm_all,
    plane_wave_sphere_radial_filters_to_order_n,
    apply_subsample_delay,
    get_fade_window,
    hrir_group_delay,
    calculate_grid_weights
)

def EndtoEnd_MagLS_weights(
    hrirs_left,
    hrirs_right,
    grid_mic_orig,
    grid_hrirs,
    fs=48000,
    c=343.0,
    order=4,
    NFFT=1024,
    f_cut=2000,
    reg_const=0.01,
    sig_len=512,
    diffuseness_constraint=False,
    diff_const_imag_thld=1e-9,
):
    """
    Compute the Magnitude Least Squares weights for a given set of HRIRs and microphone array positions.
    Parameters
    ----------
    hrirs_left : ndarray
        The left ear HRIRs.
    hrirs_right : ndarray
        The right ear HRIRs.
    grid_mic_orig : ndarray
        The microphone array positions in cartesian coordinates.
    grid_hrirs : ndarray
        The HRIR directions in cartesian coordinates.
    fs : float, optional
        The sampling frequency in Hz. The default is 48000.
    c : float, optional
        The speed of sound in m/s. The default is 343.
    order : int, optional
        The maximum order of the spherical harmonics. The default is 4.
    NFFT : int, optional
        The number of FFT points. The default is 1024.
    f_cut : int, optional
        The cut-off frequency in Hz. The default is 2000.
    reg_const : float, optional
        The regularization constant. The default is 0.01.
    sig_len : int, optional
        The length of the output filters. The default is 512.
    diffuseness_constraint : bool, optional
        Apply diffuseness constraint. The default is False.
    diff_const_imag_thld : float, optional
        The threshold for the imaginary part of the diffuseness constraint. The default is 1e-9.
    Returns
    -------
    w_mls_l : ndarray
        The left ear Magnitude Least Squares weights.
    w_mls_r : ndarray
        The right ear Magnitude Least Squares weights.

    """

    # Geometry
    az_mic, zenith_mic, r_mic = cart2sph(
        grid_mic_orig[0], grid_mic_orig[1], grid_mic_orig[2]
    )
    az_hrirs, zenith_hrirs, r_hrirs = cart2sph(
        grid_hrirs[0], grid_hrirs[1], grid_hrirs[2]
    )
    # Mic array radius
    mic_radius = r_mic[0]

    # Max SH order for plane-wave expansion: Nmax ≈ max(user_order, ka)
    Nmax = int(max(order, np.ceil(fs * np.pi * mic_radius / c)))
    N_sph_harm = (order + 1) ** 2
    freqs = np.fft.rfftfreq(NFFT, d=1 / fs)
    k = 2 * np.pi * freqs / c
    num_directions = grid_hrirs.shape[1]  # number of HRIR directions
    # Zero-pad HRIRs
    hL = np.pad(hrirs_left, ((0, NFFT - hrirs_left.shape[0]), (0, 0)), mode="constant")
    hR = np.pad(
        hrirs_right, ((0, NFFT - hrirs_right.shape[0]), (0, 0)), mode="constant"
    )

    # Group delay alignment to avoid frequency-dependent linear phase
    # grpDL = np.median(group_delay((np.sum(hL, axis=1), 1), w=freqs, fs=fs)[1])
    # grpDR = np.median(group_delay((np.sum(hR, axis=1), 1), w=freqs, fs=fs)[1])
    grpDL, grpDR = hrir_group_delay(hL, hR, fs, NFFT)

    # Time-align HRIRs
    hL = apply_subsample_delay(hL, -grpDL)
    hR = apply_subsample_delay(hR, -grpDR)

    # Frequency domain
    HL = np.fft.rfft(hL, axis=0)
    HR = np.fft.rfft(hR, axis=0)

    # Radial filters for rigid sphere (plane-wave response; minus sign matches MATLAB)
    B_N = -plane_wave_sphere_radial_filters_to_order_n(
        Nmax, k, grid_mic_orig, sphere_type="rigid"
    )

    # SH bases at mic and HRIR directions; keep first (order+1)^2 terms
    Y_NM_mic = sph_harm_all(Nmax, az_mic, zenith_mic, kind="real")
    # SMA encoder (Eq. 5): mic pressure -> SH coefficients
    Y_Lo_pinv = np.linalg.pinv(Y_NM_mic[:, :N_sph_harm])
    # SH basis at target HRTF directions
    Y_NM_hrtf = sph_harm_all(Nmax, az_hrirs, zenith_hrirs, kind="real")

    # Plane-wave encoding (forward array model)
    PNM_mic = np.einsum("kn,nr->rnk", B_N, Y_NM_mic.T, optimize=True)

    # Project to desired SH order (Eq. 5)
    PN = np.einsum("bm,mnk->bnk", Y_Lo_pinv, PNM_mic, optimize=True)

    # System transfer matrix: HRTFs in SH basis
    PNM_hrtf = np.einsum("bnk,nr->brk", PN, Y_NM_hrtf.T, optimize=True)

    # Closest frequency index to MagLS cut-off
    cut_off_indx = np.argmin(np.abs(freqs - f_cut))

    # Per-frequency SVD of SH-domain HRTFs
    U, S, Vh = np.linalg.svd(np.conj(PNM_hrtf).T, full_matrices=False)

    # Regularized inverse singular values (Eq. 11)
    max_s = np.max(S, axis=1, keepdims=True)
    regularized_S = 1.0 / np.maximum(S, reg_const * max_s)

    # Regularized pseudo-inverse: direction space -> SH weights
    Y_reg_inv = np.einsum("krb,kb,kbl->krl", U, regularized_S, Vh, optimize=True)

    # Initialize frequency-domain SH weights
    W_mls_l = np.zeros((NFFT // 2 + 1, N_sph_harm), dtype=np.complex128)
    W_mls_r = np.zeros((NFFT // 2 + 1, N_sph_harm), dtype=np.complex128)

    # Linear LS below cut-off; preserve low-frequency phase (Eq. 10)
    W_mls_l[:cut_off_indx] = np.einsum(
        "kr, krn -> kn", HL[:cut_off_indx], Y_reg_inv[:cut_off_indx], optimize=True
    )
    W_mls_r[:cut_off_indx] = np.einsum(
        "kr, krn -> kn", HR[:cut_off_indx], Y_reg_inv[:cut_off_indx], optimize=True
    )

    # Magnitude LS above cut-off with phase propagation (Eq. 14-15)
    for i in range(cut_off_indx, len(freqs)):
        phi_l = np.angle(W_mls_l[i - 1, :] @ PNM_hrtf[..., i])
        phi_h = np.angle(W_mls_r[i - 1, :] @ PNM_hrtf[..., i])
        W_mls_l[i, :] = np.einsum(
            "r, rn, r -> n", np.abs(HL[i]), Y_reg_inv[i], np.exp(1j * phi_l)
        )
        W_mls_r[i, :] = np.einsum(
            "r, rn, r -> n", np.abs(HR[i]), Y_reg_inv[i], np.exp(1j * phi_h)
        )

    # Optional diffuseness constraint (Zaunschirm): match diffuse-field covariance
    # M. Zaunschirm, C. Schorkhuber, and R. H ¨ oldrich, “Binaural rendering ¨
    # of Ambisonic signals by head-related impulse response time alignment
    # and a diffuseness constraint,” The Journal of the Acoustical Society of
    # America, vol. 143, no. 6, pp. 3616–3627, 2018.
    # The goal is to enforce that the rendered sound field has the same
    # spatial diffuseness as the original HRTFs, which can improve perceptual quality.
    if diffuseness_constraint:
        grid_weights = calculate_grid_weights(grid_hrirs)
        # Normalize grid weights to sum to 1 
        grid_weights = grid_weights / np.sum(grid_weights)
        for i in range(1, len(freqs)): # avoid zero frequency
            # Target covariance from original HRTFs (Eq. 18)
            H = np.stack((HL[i], HR[i]), axis=0)  # (2, numDirs)
            # Weighted Covariance: R = H @ W @ H^H
            # This is equivalent to H * weights @ H.conj().T
            R = (H * grid_weights) @ H.conj().T
            R = (R + R.conj().T) / 2 # Force Hermitian symmetry
            L = np.linalg.cholesky(R)

            # Rendered covariance from MagLS weights
            # The rendered HRTFs are HHat @ Y_NM_hrtf[:N_sph_harm, :].T
            HHat = np.stack((W_mls_l[i], W_mls_r[i]), axis=0)  # (2, N_sph_harm)
            H_rendered = HHat @ PNM_hrtf[..., i]

            RHat = (H_rendered * grid_weights) @ H_rendered.conj().T
            RHat = (RHat + RHat.conj().T) / 2
            LHat = np.linalg.cholesky(RHat)

            # Find the optimal mixing matrix M (Eq. 27-28)
            # This aligns the rendered covariance to the target
            # We use SVD to find the unitary part of the mapping
            U, _, Vh = np.linalg.svd(LHat.conj().T @ L)
            Q = Vh.conj().T @ U.conj().T

            # M maps the rendered HRTFs to the target covariance space
            M = L @ Q @ np.linalg.inv(LHat)
            # Apply to SH weights
            Hcorr = M @ HHat
            W_mls_l[i] = Hcorr[0]
            W_mls_r[i] = Hcorr[1]

            # Sanity check
            # Project corrected weights back to directions to compare covariance
            H_rend_corr = Hcorr @ PNM_hrtf[..., i]
            R_actual = (H_rend_corr * grid_weights) @ H_rend_corr.conj().T
            # R is the target calculated at the start of the loop
            assert np.allclose(R_actual, R, atol=1e-6)

    # Back to time domain
    w_mls_l = np.fft.irfft(W_mls_l, axis=0)
    w_mls_r = np.fft.irfft(W_mls_r, axis=0)

    # Shift to linear-phase-like alignment
    n_shift = NFFT // 2
    w_mls_l = apply_subsample_delay(w_mls_l, n_shift)
    w_mls_r = apply_subsample_delay(w_mls_r, n_shift - grpDL + grpDR)

    # Trim to target length
    start_idx = n_shift - sig_len // 2
    end_idx = n_shift + sig_len // 2
    w_mls_l = w_mls_l[start_idx:end_idx]
    w_mls_r = w_mls_r[start_idx:end_idx]

    # Apply fade window
    fade_win = get_fade_window(sig_len)
    w_mls_l = w_mls_l * fade_win[..., None]
    w_mls_r = w_mls_r * fade_win[..., None]

    return w_mls_l, w_mls_r
