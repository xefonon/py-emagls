from __future__ import annotations


import numpy as np
import soxr
from scipy.signal import group_delay, fftconvolve
from scipy.special import jv, sph_harm_y, spherical_jn, hankel2
from scipy.spatial import SphericalVoronoi, cKDTree
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def cart2sph(
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Cartesian to spherical coordinate transform.
    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates
    Returns
    -------
    theta : float or `numpy.ndarray`
            Azimuth angle in radians in [0, 2*pi)
    phi : float or `numpy.ndarray`
            Colatitude angle in radians (with 0 denoting North pole) in [0, pi]
    r : float or `numpy.ndarray`
            Radius
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)  # [0, 2*pi)
    phi = np.arccos(z / r)
    return theta, phi, r


def sph2cart(
    azimuth: float | np.ndarray,
    colatitude: float | np.ndarray,
    r: float | np.ndarray,
) -> np.ndarray:
    """Spherical to Cartesian coordinate transform.

    Parameters
    ----------
    azimuth : float or ndarray
        Azimuth angle in radians. Defined in [0, 2*pi), with 0 along x-axis and positive towards y-axis.
    colatitude : float or ndarray
        Colatitude angle in radians. Defined in [0, pi], with 0 denoting the North pole.
    r : float or ndarray
        Radius.

    Returns
    -------
    xyz : ndarray
        Cartesian coordinates with shape (3, N).
    """
    x = r * np.cos(azimuth) * np.sin(colatitude)
    y = r * np.sin(azimuth) * np.sin(colatitude)
    z = r * np.cos(colatitude)
    return np.array([x, y, z])


def get_em32_grid(mic_radius: float = 0.042) -> np.ndarray:
    """Return Eigenmike EM32 capsule positions in Cartesian coordinates.

    Parameters
    ----------
    mic_radius : float, optional
        Array radius in meters.

    Returns
    -------
    grid_mic_orig : ndarray
        Capsule positions, shape (3, 32).
    """
    theta = np.array([69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                      90.0, 125.0, 148.0, 125.0, 90.0, 55.0,
                      21.0, 58.0, 121.0, 159.0, 69.0, 90.0,
                      111.0, 90.0, 32.0, 55.0, 90.0, 125.0,
                      148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                      122.0, 159.0]) * np.pi / 180 # zenith angle (colatitude)

    phi = np.array([0.0, 32.0, 0.0, 328.0, 0.0, 45.0,
                    69.0, 45.0, 0, 315.0, 291.0, 315.0,
                    91.0, 90.0, 90.0, 89.0, 180.0, 212.0,
                    180.0, 148.0, 180.0, 225.0, 249.0, 225.0,
                    180.0, 135.0, 111.0, 135.0, 269.0, 270.0,
                    270.0, 271.0]) * np.pi / 180 # azimuth angle

    r_mic = mic_radius * np.ones_like(theta)
    # Convert from spherical to Cartesian coordinates
    return sph2cart(phi, theta, r_mic)


def mnArrays(nMax: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate degrees n and orders m up to nMax.
    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0
    Returns
    -------
    m : (int), array_like
        0, -1, 0, 1, -2, -1, 0, 1, 2, ... , -nMax ..., nMax
    n : (int), array_like
        0, 1, 1, 1, 2, 2, 2, 2, 2, ... nMax, nMax, nMax

    Notes
    -----
    Forked from https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py

    """
    # Degree n = 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
    degs = np.arange(nMax + 1)
    n = np.repeat(degs, degs * 2 + 1)

    # Order m = 0, -1, 0, 1, -2, -1, 0, 1, 2, ...
    elementNumber = np.arange((nMax + 1) ** 2) + 1
    t = np.floor(np.sqrt(elementNumber - 1)).astype(int)
    m = elementNumber - t * t - t - 1

    return m, n


def sph_harmonics(
    m: int | np.ndarray,
    n: int | np.ndarray,
    az: float | np.ndarray,
    co: float | np.ndarray,
    kind: str = "complex",
    CSphase: bool = True,
) -> np.ndarray:
    """Compute spherical harmonics.
    Parameters
    ----------
    m : (int)
        Order of the spherical harmonic. abs(m) <= n
    n : (int)
        Degree of the harmonic, sometimes called l. n >= 0
    az : (float)
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float)
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type according to complex [7]_ or
        real definition [8]_ [Default: 'complex']
    CSphase : (bool), optional
        Condon-Shortley phase convention [Default: True]

    Returns
    -------
    y_mn : (complex float) or (float)
        Spherical harmonic of order m and degree n, sampled at theta = az,
        phi = co
    References
    ----------
    .. [7] `scipy.special.sph_harm_y()`
    .. [8] Zotter, F. (2009). Analysis and Synthesis of Sound-Radiation with
        Spherical Arrays University of Music and Performing Arts Graz, Austria,
        192 pages.
    """
    # SAFETY CHECKS
    kind = kind.lower()
    if kind not in ["complex", "real"]:
        raise ValueError("Invalid kind: Choose either complex or real.")
    m = np.atleast_1d(m)

    Y = sph_harm_y(n, m, co, az)
    if not CSphase:
        CS = (-1.0) ** m
        Y /= CS
    if kind == "complex":
        return Y
    else:  # kind == 'real'
        mg0 = m > 0
        ml0 = m < 0
        Y[mg0] = np.float_power(-1.0, m)[mg0] * np.sqrt(2) * np.real(Y[mg0])
        Y[ml0] = -np.sqrt(2) * np.imag(
            Y[ml0]
        )  # negative sign applied here to match A. Politis implementation
        return np.real(Y)


def sph_harm_all(
    nMax: int,
    az: np.ndarray,
    co: np.ndarray,
    kind: str = "complex",
    CSphase: bool = True,
) -> np.ndarray:
    """Compute all spherical harmonic coefficients up to degree nMax.
    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0
    az: (float), array_like
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float), array_like
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    Returns
    -------
    y_mn : (complex float) or (float), array_like
        Spherical harmonics of degrees n [0 ... nMax] and all corresponding
        orders m [-n ... n], sampled at [az, co]. dim1 corresponds to az/co
        pairs, dim2 to oder/degree (m, n) pairs like 0/0, -1/1, 0/1, 1/1,
        -2/2, -1/2 ...
    """
    m, n = mnArrays(nMax)
    mA, azA = np.meshgrid(m, az)
    nA, coA = np.meshgrid(n, co)
    return sph_harmonics(mA, nA, azA, coA, kind=kind, CSphase=CSphase)


def scalar_broadcast_match(
    a: float | np.ndarray,
    b: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns arguments as np.array, if one is a scalar it will broadcast the
    other one's shape.
    """
    a, b = np.atleast_1d(a, b)
    if a.size == 1 and b.size != 1:
        a = np.broadcast_to(a, b.shape)
    elif b.size == 1 and a.size != 1:
        b = np.broadcast_to(b, a.shape)
    return a, b


def besselj(n: int | np.ndarray, z: float | np.ndarray) -> np.ndarray:
    """Bessel function of first kind of order n at kr. Wraps
    `scipy.special.jn(n, z)`.
    Parameters
    ----------
    n : array_like
        Order
    z: array_like
        Argument
    Returns
    -------
    J : array_like
        Values of Bessel function of order n at position z
    """
    return jv(n, np.complex128(z))


def spbessel(n: int | np.ndarray, kr: float | np.ndarray) -> np.ndarray:
    r"""Spherical Bessel function (first kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    J : complex float
        Spherical Bessel
    """
    n, kr = scalar_broadcast_match(n, kr)

    if np.any(n < 0) | np.any(kr < 0) | np.any(np.mod(n, 1) != 0):
        J = np.zeros(kr.shape, dtype=np.complex128)

        kr_non_zero = kr != 0
        J[kr_non_zero] = np.lib.scimath.sqrt(np.pi / 2 / kr[kr_non_zero]) * besselj(
            n[kr_non_zero] + 0.5, kr[kr_non_zero]
        )
        J[np.logical_and(kr == 0, n == 0)] = 1
    else:
        J = spherical_jn(n.astype(int), kr)
    return np.squeeze(J)


def sphankel2(n: int | np.ndarray, kr: float | np.ndarray) -> np.ndarray:
    r"""Spherical Hankel (second kind) of order n at kr.
    Parameters
    ----------
    n : array_like
       Order
    kr: array_like
       Argument
    Returns
    -------
    hn2 : complex float
       Spherical Hankel function hn (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    hn2 = np.full(n.shape, np.nan, dtype=np.complex128)
    kr_nonzero = kr != 0
    hn2[kr_nonzero] = (
        np.sqrt(np.pi / 2)
        / np.lib.scimath.sqrt(kr[kr_nonzero])
        * hankel2(n[kr_nonzero] + 0.5, kr[kr_nonzero])
    )
    return hn2


def dspbessel(n: int | np.ndarray, kr: float | np.ndarray) -> np.ndarray:
    """Derivative of spherical Bessel (first kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    J' : complex float
        Derivative of spherical Bessel
    """
    return np.squeeze(
        (n * spbessel(n - 1, kr) - (n + 1) * spbessel(n + 1, kr)) / (2 * n + 1)
    )


def dsphankel2(n: int | np.ndarray, kr: float | np.ndarray) -> np.ndarray:
    """Derivative spherical Hankel (second kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    dhn2 : complex float
        Derivative of spherical Hankel function hn' (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    dhn2 = np.full(n.shape, np.nan, dtype=np.complex128)
    kr_nonzero = kr != 0
    dhn2[kr_nonzero] = 0.5 * (
        sphankel2(n[kr_nonzero] - 1, kr[kr_nonzero])
        - sphankel2(n[kr_nonzero] + 1, kr[kr_nonzero])
        - sphankel2(n[kr_nonzero], kr[kr_nonzero]) / kr[kr_nonzero]
    )
    return dhn2


def bn_rigid_omni(
    n: int,
    kr: float | np.ndarray,
    ka: float | np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Radial function for scattering of rigid sphere
    Parameters
    ----------
    n: order of radial functions
    kr: Helmholtz number for evaluation radius (scalar or array-like)
    ka: Helmholtz number for sphere radius (scalar or array-like)

    Returns
    -------
    radial function for rigid sphere
    """
    if normalize:
        scale_factor = np.squeeze(4 * np.pi * 1j**n)
    else:
        scale_factor = 1.0
    kr, ka = scalar_broadcast_match(kr, ka)
    result = np.zeros_like(kr, dtype=complex)
    
    nonzero = (kr != 0) & (ka != 0)
    result[nonzero] = scale_factor * (
        spbessel(n, kr[nonzero]) - 
        (dspbessel(n, ka[nonzero]) / dsphankel2(n, ka[nonzero])) * sphankel2(n, kr[nonzero])
    )
    
    # Analytical DC limit
    if np.any(~nonzero):
        if n == 0:
            result[~nonzero] = scale_factor * 1.0  # 4π * i^0 * 1 = 4π
        else:
            result[~nonzero] = 0.0
    
    return result


def bn_open_omni(n: int, kr: float | np.ndarray, normalize: bool = False) -> np.ndarray:
    """Radial function for an open (non-rigid) sphere.

    Parameters
    ----------
    n: order of radial functions
    kr: Helmholtz number for evaluation radius (scalar or array-like)
    normalize: bool, optional
        Whether to apply normalization factor (4 * pi * 1j^n). Default is False.
    """
    if normalize:
        scale_factor = np.squeeze(4 * np.pi * 1j**n)
    else:
        scale_factor = 1.0

    return scale_factor * spherical_jn(n, kr)


def get_radial_filters_to_order_n(
    order_max: int,
    kr: np.ndarray,
    ka: np.ndarray,
    sphere_type: str = "rigid",
    normalize: bool = True,
) -> np.ndarray:
    """Compute radial filters up to order max for a given sphere type.

    Parameters
    ----------
    order_max: int
        Maximum order of radial filters to compute.
    kr: np.ndarray
        Helmholtz number for evaluation radius (shape: [n_freqs, n_mics])
    ka: np.ndarray
        Helmholtz number for sphere radius (shape: [n_freqs, n_mics])
    sphere_type: str, optional
        Type of sphere ('rigid' or 'open'). Default is 'rigid'."""
    if sphere_type == "rigid":
        func = lambda n, kr, ka: bn_rigid_omni(n, kr, ka, normalize) # noqa: E731
    else:
        func = lambda n, kr, ka: bn_open_omni(n, kr, normalize) # noqa: E731
    nold = 0
    NMLocatorSize = (order_max + 1) ** 2
    radial_filters = np.zeros((kr.shape[0], kr.shape[1], NMLocatorSize), dtype=complex)

    for n in range(0, order_max + 1):
        bn = func(n, kr, ka)
        nnew = nold + 2 * n + 1
        radial_filters[:, :, nold:nnew] = bn[..., None]
        nold = nnew
    return radial_filters


def plane_wave_sphere_radial_filters_to_order_n(
    order_max: int,
    k: np.ndarray,
    sphere_grid: np.ndarray,
    sphere_type: str = "rigid",
    c: float = 343.0,
    delay: float = 0.0,
) -> np.ndarray:
    """Compute radial filters for a plane wave incident on a sphere.
        For reference, see [1]_.
    Parameters
    ----------
    order_max : int
        Maximum order of the spherical harmonics.
    k : array_like
        Wavenumber.
    sphere_grid : array_like
        Grid of the sphere.
    sphere_type : str
        Type of the sphere.
    c : float
        Speed of sound.
    delay : float
        Delay of the plane wave.
    Returns
    -------
    radial_filters : array_like
        Radial filters of shape (N_freqs, N_radial_filters).

    [1] Rafaely, B. (2015). Fundamentals of spherical array processing. Springer.
    """
    freqs = k * c / (2 * np.pi)
    omega = 2 * np.pi * freqs
    time_shift = np.exp(-1j * omega * delay)
    
    R = np.linalg.norm(sphere_grid[:,0])  # or pass mic_radius directly
    ka = k[None, :] * R
    kr = ka.copy()

    # NMLocatorSize = (order_max + 1) ** 2
    B_N = get_radial_filters_to_order_n(
        order_max, kr, ka, sphere_type, normalize=True
    ).squeeze(0)

    return B_N * time_shift[:, None]

def _next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())

def plane_waves(
    n0: np.ndarray,
    freqs: np.ndarray,
    grid: np.ndarray,
    c: float = 343.0,
    fs: int = 48000,
    safety: int = 256,        # extra guard samples for sinc decay
    return_length: int | None = None,
):
    """
    Time-domain causal plane-wave impulse responses.

    Parameters
    ----------
    n0 : (3,) or (3, N)
        Plane-wave direction vectors.
    freqs : (F,)
        Positive rfft frequencies (Hz) including DC and Nyquist.
    grid : (3, M)
        Microphone cartesian positions (meters).
    c : float
        Speed of sound.
    safety : int
        Extra padding to suppress circular wrap.
    return_length : int or None
        If given, crop the causal part to this length.

    Returns
    -------
    h : (T, M) or (T, M, N)
        Causal impulse responses.
    """

    n0 = np.asarray(n0, float)
    freqs = np.asarray(freqs, float)
    grid = np.asarray(grid, float)

    if n0.ndim == 1:
        n0 = n0[:, None]

    # Geometric delays for plane wave at each microphone position, 
    # relative to the earliest arrival
    tau = (grid.T @ n0) / c           # (M,N)
    tau -= tau.min(axis=0, keepdims=True)

    max_delay = tau.max()
    max_delay_samples = int(np.ceil(max_delay * fs))

    # Choose FFT length to avoid circular convolution wrap, with some 
    # extra safety margin for the sinc decay
    nfft_nominal = 2 * (len(freqs) - 1)
    nfft = _next_pow2(nfft_nominal + max_delay_samples + safety)

    # recompute frequency grid to match new FFT
    freqs_full = np.fft.rfftfreq(nfft, 1/fs)

    omega = 2 * np.pi * freqs_full[:, None, None]
    H = np.exp(-1j * omega * tau[None, :, :])

    h = np.fft.irfft(H, n=nfft, axis=0)

    # Crop causal useful part
    if return_length is None:
        return_length = max_delay_samples + safety

    h = h[:return_length]

    return np.squeeze(h)

def fibonacci_grid_sphere(n_points: int) -> np.ndarray:
    """
    Generate a Fibonacci grid on the sphere.

    Parameters
    ----------
    n_points : int
        Number of points in the grid.

    Returns
    -------
    grid : (3, n_points) ndarray
        Cartesian coordinates of the grid points on the sphere.
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    # radius is 1 for unit sphere
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
   
    return np.array([x, y, z])

def calculate_smr(
    w_mls_l: np.ndarray,
    w_mls_r: np.ndarray,
    w_ref_l: np.ndarray,
    w_ref_r: np.ndarray,
    fs: int,
    order_mls: int,
    order_ref: int,
    NFFT: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Summed Magnitude Response (SMR) for MagLS

    This corresponds to Eq. 16 in the paper. [1]

    Parameters
    ----------
    w_mls_l, w_mls_r: (NFFT//2 + 1, SH_channels) ndarray
        Magnitude Least Squares weights for left and right ears.
    w_ref_l, w_ref_r: (NFFT//2 + 1, SH_channels) ndarray
        Reference weights for left and right ears.
    fs: int
        Sampling frequency.
    order_mls: int
        Spherical harmonic order used for MagLS method.
    order_ref: int
        Spherical harmonic order of the reference grid
    NFFT: int
        FFT length used for frequency analysis.

    Returns
    -------
    freqs: (F,) ndarray
        Frequency bins corresponding to the SMR values.
    smr_mls_db: (F, n_dirs) ndarray
        SMR of the MagLS method in dB.
    smr_ref_db: (F, n_dirs) ndarray
        SMR of the reference method in dB.
    smr_err_db: (F, n_dirs) ndarray
        Difference in SMR between MagLS and reference in dB.

    References
    ----------
    [1] "End-to-End Magnitude Least Squares Binaural Rendering of 
        Spherical Microphone Array Recordings" Deppisch et al., 
        Immersive and 3D Audio: from Architecture to Automotive (I3DA), 2021.
    """
    W_mls_l = np.fft.rfft(w_mls_l, n=NFFT, axis=0)  # (F, Nm)
    W_mls_r = np.fft.rfft(w_mls_r, n=NFFT, axis=0)
    W_ref_l = np.fft.rfft(w_ref_l, n=NFFT, axis=0)  # (F, Nr)
    W_ref_r = np.fft.rfft(w_ref_r, n=NFFT, axis=0)

    fib_sphere = fibonacci_grid_sphere(2500)

    azi, zen, _ = cart2sph(fib_sphere[0], fib_sphere[1], fib_sphere[2])
    # SH basis for MLS order and reference order (e.g. plane wave responses)
    Y_mls = sph_harm_all(order_mls, azi, zen, kind="real")  # (K, Nm)
    
    Y_ref = sph_harm_all(order_ref, azi, zen, kind="real")  # (K, Nr)
    n_dirs = len(azi)
    # Reconstructed binaural transfer functions H(ω, Ω)
    H_mls_l = np.einsum("fn,kn->fk", W_mls_l, Y_mls, optimize=True)
    H_mls_r = np.einsum("fn,kn->fk", W_mls_r, Y_mls, optimize=True)
    H_ref_l = np.einsum("fn,kn->fk", W_ref_l, Y_ref, optimize=True)
    H_ref_r = np.einsum("fn,kn->fk", W_ref_r, Y_ref, optimize=True)

    smr_mls_db = 10.0 * np.log10(0.5 * (np.abs(H_mls_l) ** 2 + np.abs(H_mls_r) ** 2) + 1e-12)
    smr_ref_db = 10.0 * np.log10(0.5 * (np.abs(H_ref_l) ** 2 + np.abs(H_ref_r) ** 2) + 1e-12)

    # Eq. (16)
    smr_err_db = smr_mls_db - smr_ref_db
    freqs = np.fft.rfftfreq(NFFT, d=1 / fs)
    return freqs, smr_mls_db, smr_ref_db, smr_err_db

def plot_smr_err_freq_vs_azimuth(
    freqs: np.ndarray,
    smr_err_db: np.ndarray,   # shape (F, K)
    grid_cart: np.ndarray,    # shape (3, K)
    n_az_bins: int = 180,
    horiz_only: bool = False,
    zen_tol_deg: float = 10.0,
    clim: tuple[float, float] = (-15, 15),
    title: str = "SMR error (dB): freq vs azimuth",
):
    azi, zen, _ = cart2sph(grid_cart[0], grid_cart[1], grid_cart[2])
    az_deg = (np.rad2deg(azi) + 360.0) % 360.0
    zen_deg = np.rad2deg(zen)

    # Optional: only directions near horizontal plane (zenith ~ 90 deg)
    dir_mask = np.ones_like(az_deg, dtype=bool)
    if horiz_only:
        dir_mask = np.abs(zen_deg - 90.0) <= zen_tol_deg

    az_edges = np.linspace(0.0, 360.0, n_az_bins + 1)
    img = np.full((len(freqs), n_az_bins), np.nan)

    for b in range(n_az_bins):
        m = (
            (az_deg >= az_edges[b]) &
            (az_deg < az_edges[b + 1]) &
            dir_mask
        )
        if np.any(m):
            img[:, b] = np.nanmean(smr_err_db[:, m], axis=1)

    # log-frequency plotting: drop DC
    f = freqs[1:]
    z = img[1:, :]  # (freq, az_bin)

    plt.figure(figsize=(10, 5))
    pcm = plt.pcolormesh(
        az_edges[:-1],  # x: azimuth
        f,              # y: frequency
        z,
        shading="auto",
        cmap="RdBu_r",
        vmin=clim[0],
        vmax=clim[1],
    )
    # plt.yscale("log")
    plt.ylim([max(20, f.min()), f.max()])
    plt.xlim([0, 360])
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(pcm, label="SMR error (dB)")
    plt.tight_layout()
    plt.savefig(f"smr_err_freq_vs_az_{title.replace(' ', '_')}.png")

def plot_diffuse_field_validation(
    w_mls_l: np.ndarray,
    w_mls_r: np.ndarray,
    w_ls_l: np.ndarray,
    w_ls_r: np.ndarray,
    hrirs_left: np.ndarray,
    hrirs_right: np.ndarray,
    hrir_grid_cart: np.ndarray,
    fs: int,
    order: int,
    title: str,
    NFFT: int = 1024,
):
    """
    Validates the diffuse-field magnitude response approximation (Eq 19 & 20).
    """
    W_mls_l = np.fft.rfft(w_mls_l, n=NFFT, axis=0)  # [NFFT//2 + 1, SH_channels]
    W_mls_r = np.fft.rfft(w_mls_r, n=NFFT, axis=0)
    W_ls_l = np.fft.rfft(w_ls_l, n=NFFT, axis=0)  # [NFFT//2 + 1, SH_channels]
    W_ls_r = np.fft.rfft(w_ls_r, n=NFFT, axis=0)
    n_freqs = W_mls_l.shape[0]
    n_dirs = hrirs_left.shape[1]

    # Calculate weights (alpha_k) for the grid (assuming Voronoi for irregular grids)
    # If the grid is nearly uniform, alpha_k can be 1/K
    # Normalize grid points to lie on the unit sphere for Voronoi area calculation
    hrir_grid_cart = hrir_grid_cart
    alpha = calculate_grid_weights(hrir_grid_cart)  # [K,]

    # Target Diffuse-Field Response |H_df(w)| (Eq 19)
    # Average of magnitude responses across both ears and all directions
    H_df_target = np.zeros(n_freqs)
    for k in range(n_dirs):
        mag_l = np.abs(np.fft.rfft(hrirs_left[:, k], n=NFFT))[:n_freqs]
        mag_r = np.abs(np.fft.rfft(hrirs_right[:, k], n=NFFT))[:n_freqs]
        H_df_target += alpha[k] * 0.5 * (mag_l + mag_r)

    # Rendered Diffuse-Field Response |H_df,MLS(w)| (Eq 20)
    # Uses the pressure vector p^H(w, Omega_k) and MagLS weights
    fib_sphere = fibonacci_grid_sphere(2500)

    azi, zen, _ = cart2sph(fib_sphere[0], fib_sphere[1], fib_sphere[2])
    Y_grid = sph_harm_all(order, azi, zen, kind="real")  # [K, (N+1)**2]
    alpha = calculate_grid_weights(fib_sphere)  # [K,]
    n_dirs = Y_grid.shape[0]
    H_df_mls = np.zeros(n_freqs)
    for k in range(n_dirs):
        # p^H * w is effectively the reconstruction at that direction
        rec_mag_l = np.abs(W_mls_l @ Y_grid[k, :])
        rec_mag_r = np.abs(W_mls_r @ Y_grid[k, :])
        H_df_mls += alpha[k] * 0.5 * (rec_mag_l + rec_mag_r)
    H_df_ls = np.zeros(n_freqs)
    for k in range(n_dirs):
        rec_mag_l = np.abs(W_ls_l @ Y_grid[k, :])
        rec_mag_r = np.abs(W_ls_r @ Y_grid[k, :])
        H_df_ls += alpha[k] * 0.5 * (rec_mag_l + rec_mag_r)

    # normalise by 4π
    H_df_mls *= np.sqrt(4 * np.pi)
    H_df_ls *= np.sqrt(4 * np.pi)

    freqs = np.fft.rfftfreq(NFFT, 1 / fs)[:n_freqs]
    # normalize magnitudes by m
    plt.figure(figsize=(10, 5))
    plt.semilogx(freqs, 20 * np.log10(H_df_target), "k", label="Target |H_df| (Eq 19)")
    plt.semilogx(freqs, 20 * np.log10(H_df_mls), "r--", label="MagLS |H_df,MLS| (Eq 20)")
    plt.semilogx(freqs, 20 * np.log10(H_df_ls), "b-.", label="LS |H_df,LS|")

    plt.title(f"Diffuse-Field Magnitude Reconstruction - {title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.xlim([100, fs / 2])
    plt.ylim([-5, 40])
    plt.savefig(f"diffuse_field_val_{title.replace(' ', '_')}.png")

def get_fade_window(
    ir_len: int, rel_fade_len: float = 0.15, window_type: str = "hann"
) -> np.ndarray:
    """
    Return combined fade-in and fade-out window.

    Parameters:
    ir_len (int): Length of impulse response.
    rel_fade_len (float): Relative length of fade-in and fade-out windows.

    Returns:
    numpy.ndarray: Fade window.
    """

    n_fadein = round(rel_fade_len * ir_len)
    n_fadeout = round(rel_fade_len * ir_len)

    if window_type == "raised_cosine":
        # Smoother than Hanning
        fadein = 0.5 * (1 - np.cos(np.pi * np.arange(n_fadein) / n_fadein))
        fadeout = 0.5 * (1 + np.cos(np.pi * np.arange(n_fadeout) / n_fadeout))
    else:  # hann
        hannin = np.hanning(2 * n_fadein)
        fadein = hannin[:n_fadein]
        hannout = np.hanning(2 * n_fadeout)
        fadeout = hannout[n_fadeout:]

    fade_win = np.concatenate(
        [
            fadein,
            np.ones(ir_len - (n_fadein + n_fadeout)),
            fadeout,
        ]
    )

    return fade_win


def apply_subsample_delay(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Apply a time delay with sub-sample precision to an input signal.

    Parameters:
    sig (numpy.ndarray): Input signal array.
    delay_samples (float): The delay in samples.

    Returns:
    numpy.ndarray: Delayed signal.
    """
    sig = np.asarray(sig)
    n = sig.shape[0]

    # one-sided spectrum
    S = np.fft.rfft(sig, axis=0)
    k = np.arange(S.shape[0], dtype=float)  # bin index
    phase = np.exp(-1j * 2.0 * np.pi * k * delay_samples / n)[:, None]

    # Nyquist should be real for even n
    if n % 2 == 0:
        phase[-1] = np.real(phase[-1])

    y = np.fft.irfft(S * phase, n=n, axis=0)
    return y


def hrir_group_delay(
    hrirs_l: np.ndarray,
    hrirs_r: np.ndarray,
    fs: int,
    NFFT: int,
) -> tuple[float, float]:
    """
    Estimate the group delay of a set of HRIRs.

    Parameters:
    hrirs (numpy.ndarray): Array of shape (N_samples, N_hrirs) containing the HRIRs.
    fs (int): Sampling frequency in Hz.
    NFFT (int): Number of FFT points to use for the group delay estimation.

    Returns:
    numpy.ndarray: Array of shape (N_hrirs,) containing the estimated group delays in seconds.
    """
    freqs = np.fft.rfftfreq(NFFT, d=1 / fs)

    hL_sum = np.sum(hrirs_l, axis=1)  # sum over directions
    hR_sum = np.sum(hrirs_r, axis=1)

    _, gd_l = group_delay((hL_sum, np.array([1.0])), w=freqs, fs=fs)
    _, gd_r = group_delay((hR_sum, np.array([1.0])), w=freqs, fs=fs)

    gd_l = gd_l[np.isfinite(gd_l)]
    gd_r = gd_r[np.isfinite(gd_r)]

    grpDL = float(np.median(gd_l))
    grpDR = float(np.median(gd_r))
    return grpDL, grpDR


def binaural_decode(
    ambisonic_signals: np.ndarray,
    input_fs: int,
    decode_filter_left: np.ndarray,
    decode_filter_right: np.ndarray,
    decode_filter_fs: int,
    compensate_delay: bool = False,
    mono_signal: np.ndarray | None = None,
    mono_signal_fs: int | None = None,
    hor_rotation_angle_rad: float | None = None,
) -> np.ndarray:
    """
    Decode SH-domain (Ambisonic) signal to binaural, optionally convolve
    with a mono signal (in case ambisonic signal is a RIR).

    Parameters:
    ambisonic_signals : np.ndarray
        Ambisonic signals (SH-domain), shape (num_samples, num_harmonics)
    input_fs : int
        Sampling rate of input Ambisonic signals
    decode_filter_left : np.ndarray
        Left ear decoding filters, shape (num_samples_filter, num_harmonics)
    decode_filter_right : np.ndarray
        Right ear decoding filters, shape (num_samples_filter, num_harmonics)
    decode_filter_fs : int
        Sampling rate of decoding filters
    compensate_delay : bool, optional
        Whether to compensate for delay (default: False)
    mono_signal : np.ndarray, optional
        Optional mono signal to convolve with binaural output, shape (num_samples_signal, 1)
    mono_signal_fs : int, optional
        Sampling rate of optional mono signal
    hor_rotation_angle_rad : float, optional
        Horizontal rotation angle in radians (default: None)

    Returns:
    np.ndarray
        Binaural output, shape (num_samples, 2) for left and right channels
    """

    # Resample mono signal if necessary
    if mono_signal is not None and mono_signal_fs is not None:
        if mono_signal_fs != input_fs:
            print("binaural_decode: resampling mono signal")
            if mono_signal.ndim == 1:
                mono_signal = soxr.resample(mono_signal, mono_signal_fs, input_fs)
            else:
                mono_signal = np.stack(
                    [
                        soxr.resample(mono_signal[:, i], mono_signal_fs, input_fs)
                        for i in range(mono_signal.shape[1])
                    ],
                    axis=-1,
                )

    # Resample decoding filters if necessary
    if decode_filter_fs != input_fs:
        print("binaural_decode: resampling decoding filters")
        decode_filter_left = np.stack(
            [
                soxr.resample(decode_filter_left[:, i], decode_filter_fs, input_fs)
                for i in range(decode_filter_left.shape[1])
            ],
            axis=-1,
        )
        decode_filter_right = np.stack(
            [
                soxr.resample(decode_filter_right[:, i], decode_filter_fs, input_fs)
                for i in range(decode_filter_right.shape[1])
            ],
            axis=-1,
        )

    # Apply horizontal rotation if specified
    if hor_rotation_angle_rad is not None and hor_rotation_angle_rad != 0:
        # TODO: This function is untested
        # convert hor_rotation_angle_rad to yaw, pitch, roll
        ambisonic_signals = rotate_sh(ambisonic_signals, hor_rotation_angle_rad, 0, 0)

    # Ambisonic to binaural decoding
    num_samples, num_harmonics = ambisonic_signals.shape
    left_ear_signal = np.zeros(num_samples)
    right_ear_signal = np.zeros(num_samples)

    for i in range(num_harmonics):
        left_ear_signal += fftconvolve(
            ambisonic_signals[:, i], decode_filter_left[:, i], mode="full"
        )[:num_samples]
        right_ear_signal += fftconvolve(
            ambisonic_signals[:, i], decode_filter_right[:, i], mode="full"
        )[:num_samples]

    # Optionally convolve with mono signal
    if mono_signal is not None:
        left_ear_signal = fftconvolve(left_ear_signal, mono_signal[:, 0], mode="full")[
            :num_samples
        ]
        right_ear_signal = fftconvolve(
            right_ear_signal, mono_signal[:, 0], mode="full"
        )[:num_samples]

    # Prepare output
    binaural_output = np.column_stack((left_ear_signal, right_ear_signal))

    # Compensate delay if specified
    if compensate_delay:
        delay = len(decode_filter_left) // 2
        binaural_output = binaural_output[delay:, :]

    # Ensure the output is real
    if not np.isreal(binaural_output).all():
        imaginary_sum = (
            np.sum(np.abs(binaural_output[:, 0].imag)),
            np.sum(np.abs(binaural_output[:, 1].imag)),
        )
        print(
            "Warning: Discarding imaginary part with sum of "
            f"[{imaginary_sum[0]:.2g}, {imaginary_sum[1]:.2g}] in rendering result."
        )
        binaural_output = np.real(binaural_output)

    return binaural_output


def sh_rotation_matrix(N_sph: int, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Stabilized real-valued SH rotation matrix (Wigner-D).
    Fixes the HF coloration by ensuring strict orthogonality and correct ACN mapping.
    """
    # 1. Use Scipy to get a clean, orthogonal 3x3 rotation matrix
    # Standard 'zyx' convention (Yaw-Pitch-Roll)
    r_obj = R.from_euler("zyx", [yaw, pitch, roll])
    r3x3 = r_obj.as_matrix()

    # 2. Map Scipy's [X, Y, Z] to Ambisonic ACN Order 1: [Y, Z, X]
    # This mapping is the most common cause of 'crazy coloration'
    r1 = np.zeros((3, 3))
    reorder = [1, 2, 0]  # Map Y(1), Z(2), X(0)
    for i in range(3):
        for j in range(3):
            r1[i, j] = r3x3[reorder[i], reorder[j]]

    r_blocks = [np.array([[1.0]])]  # Order 0 (Omni) is always 1
    r_blocks.append(r1)  # Order 1 (Dipoles)

    # Ivanic Recursion Helpers
    def P(i, a, b, ell, r1, rlm1):
        if b == -ell:
            return (
                r1[i + 1, 2] * rlm1[a + ell - 1, 0]
                + r1[i + 1, 0] * rlm1[a + ell - 1, 2 * ell - 2]
            )
        if b == ell:
            return (
                r1[i + 1, 2] * rlm1[a + ell - 1, 2 * ell - 2]
                - r1[i + 1, 0] * rlm1[a + ell - 1, 0]
            )
        else:
            return r1[i + 1, 1] * rlm1[a + ell - 1, b + ell - 1]

    def U(ell, m, n, r1, rlm1):
        return P(0, m, n, ell, r1, rlm1)

    def V(ell, m, n, r1, rlm1):
        if m == 0:
            return P(1, 1, n, ell, r1, rlm1) + P(-1, -1, n, ell, r1, rlm1)
        if m > 0:
            res = P(1, m - 1, n, ell, r1, rlm1)
            if m == 1:
                return res * np.sqrt(2)
            return res - P(-1, 1 - m, n, ell, r1, rlm1)
        res = P(-1, -m - 1, n, ell, r1, rlm1)
        if m == -1:
            return res * np.sqrt(2)
        return res + P(1, m + 1, n, ell, r1, rlm1)

    def W(ell, m, n, r1, rlm1):
        if m > 0:
            return P(1, m + 1, n, ell, r1, rlm1) + P(
                -1, -m - 1, n, ell, r1, rlm1
            )
        if m < 0:
            return P(1, m - 1, n, ell, r1, rlm1) - P(
                -1, 1 - m, n, ell, r1, rlm1
            )
        return 0

    # 3. Compute higher orders (N=2, 3...)
    for ell in range(2, N_sph + 1):
        prev_block = r_blocks[ell - 1]
        this_block = np.zeros((2 * ell + 1, 2 * ell + 1))
        for m in range(-ell, ell + 1):
            for n in range(-ell, ell + 1):
                d = 1 if m == 0 else 0
                if abs(n) == ell:
                    denom = 2 * ell * (2 * ell - 1)
                else:
                    denom = ell * ell - n * n
                u = np.sqrt((ell * ell - m * m) / denom)
                v = (
                    0.5
                    * np.sqrt((1 + d) * (ell + abs(m) - 1) * (ell + abs(m)) / denom)
                    * (1 - 2 * d)
                )
                w = (
                    -0.5
                    * np.sqrt((ell - abs(m) - 1) * (ell - abs(m)) / denom)
                    * (1 - d)
                )

                res = 0
                if u != 0:
                    res += u * U(ell, m, n, r1, prev_block)
                if v != 0:
                    res += v * V(ell, m, n, r1, prev_block)
                if w != 0:
                    res += w * W(ell, m, n, r1, prev_block)
                this_block[m + ell, n + ell] = res
        r_blocks.append(this_block)

    # Assemble into a giant block-diagonal matrix
    dim = (N_sph + 1) ** 2
    R_full = np.zeros((dim, dim))
    idx = 0
    for block in r_blocks:
        s = block.shape[0]
        R_full[idx : idx + s, idx : idx + s] = block
        idx += s
    return R_full

def rotate_sh(F_nm, yaw, pitch, roll, is_head_tracking=True):
    """
    Applies the rotation matrix to the SH signals.
    """
    # For head tracking, we rotate the sound field OPPOSITE to the head
    if is_head_tracking:
        yaw, pitch, roll = -yaw, -pitch, -roll
        
    N_sph = int(np.sqrt(F_nm.shape[1]) - 1)
    rot_matrix = sh_rotation_matrix(N_sph, yaw, pitch, roll)
    
    # Perform the rotation in the SH domain
    return F_nm @ rot_matrix.T

def rotation_euler(yaw: float = 0, pitch: float = 0, roll: float = 0) -> np.ndarray:
    """Matrix rotating by Yaw (around z), pitch (around y), roll (around x).
    See https://mathworld.wolfram.com/RotationMatrix.html
    """
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx

def _unit_normal(a, b, c):
    """
    Forked from Spharpy, see
    https://github.com/pyfar/spharpy/blob/017337084074f030f0fdc5f0bd07c83562e9a709/spharpy/samplings/helpers.py
    """
    x = np.linalg.det(
        [[1, a[1], a[2]],
         [1, b[1], b[2]],
         [1, c[1], c[2]]])
    y = np.linalg.det(
        [[a[0], 1, a[2]],
         [b[0], 1, b[2]],
         [c[0], 1, c[2]]])
    z = np.linalg.det(
        [[a[0], a[1], 1],
         [b[0], b[1], 1],
         [c[0], c[1], 1]])

    magnitude = np.sqrt(x**2 + y**2 + z**2)

    return (x/magnitude, y/magnitude, z/magnitude)

def _poly_area(poly):
    """
    Forked from Spharpy, see

    https://github.com/pyfar/spharpy/blob/017337084074f030f0fdc5f0bd07c83562e9a709/spharpy/samplings/helpers.py
    """
    # area of polygon poly
    if len(poly) < 3:
        # not a plane - no area
        return 0
    total = [0.0, 0.0, 0.0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[np.mod((i+1), N)]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, _unit_normal(poly[0], poly[1], poly[2]))
    return np.abs(result/2)

def calculate_grid_weights(grid_cart):
    """
    Computes quadrature weights for an arbitrary grid on a sphere.
    grid_cart : (3, N) array of cartesian coordinates.
    Returns: (N,) array of weights summing to 4*pi.
    """
    # Ensure points are on a unit sphere
    points = grid_cart.T
    n_points = points.shape[0]
    radius = np.linalg.norm(points, axis=1).mean()
    # Compute Voronoi regions
    # Note: Requires points to be unique and generally not all on one plane
    sv = SphericalVoronoi(points, radius=radius, center = 0.)
    
    sv.sort_vertices_of_regions()

    unique_verts, idx_uni = np.unique(
        np.round(sv.vertices, decimals=10),
        axis=0,
        return_index=True)
    
    searchtree = cKDTree(unique_verts)
    area = np.zeros(n_points, float)

    for idx, region in enumerate(sv.regions):
        _, idx_nearest = searchtree.query(sv.vertices[np.array(region)])
        mask_unique = np.sort(np.unique(idx_nearest, return_index=True)[1])
        mask_new = idx_uni[idx_nearest[mask_unique]]

        area[idx] = _poly_area(sv.vertices[mask_new])

    area = area / np.sum(area) * 4 * np.pi

    return area