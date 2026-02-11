from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import soxr
from scipy.signal import group_delay, fftconvolve
from scipy.special import jv, sph_harm_y, spherical_jn, hankel2
from scipy.spatial import SphericalVoronoi

def cart2sph(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            Azimuth angle in radians
    phi : float or `numpy.ndarray`
            Colatitude angle in radians (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return theta, phi, r


def sph2cart(
    azimuth: Union[float, np.ndarray],
    colatitude: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
) -> np.ndarray:
    """Spherical to Cartesian coordinate transform.

    Parameters
    ----------
    azimuth : float or ndarray
        Azimuth angle in radians.
    colatitude : float or ndarray
        Colatitude angle in radians.
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
    em32_azi_deg = np.array(
        [
            0,
            32,
            0,
            328,
            0,
            45,
            69,
            45,
            0,
            315,
            291,
            315,
            91,
            90,
            90,
            89,
            180,
            212,
            180,
            148,
            180,
            225,
            249,
            225,
            180,
            135,
            111,
            135,
            269,
            270,
            270,
            271,
        ]
    )

    em32_zen_deg = np.array(
        [
            69,
            90,
            111,
            90,
            32,
            55,
            90,
            125,
            148,
            125,
            90,
            55,
            21,
            58,
            121,
            159,
            69,
            90,
            111,
            90,
            32,
            55,
            90,
            125,
            148,
            125,
            90,
            55,
            21,
            58,
            122,
            159,
        ]
    )

    az_mic = np.deg2rad(em32_azi_deg)
    zenith_mic = np.deg2rad(em32_zen_deg)
    r_mic = mic_radius * np.ones_like(az_mic)
    return sph2cart(az_mic, zenith_mic, r_mic)


def mnArrays(nMax: int) -> Tuple[np.ndarray, np.ndarray]:
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
    m: Union[int, np.ndarray],
    n: Union[int, np.ndarray],
    az: Union[float, np.ndarray],
    co: Union[float, np.ndarray],
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

    Y = sph_harm_y(m, n, az, co)
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
    a: Union[float, np.ndarray], b: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns arguments as np.array, if one is a scalar it will broadcast the
    other one's shape.
    """
    a, b = np.atleast_1d(a, b)
    if a.size == 1 and b.size != 1:
        a = np.broadcast_to(a, b.shape)
    elif b.size == 1 and a.size != 1:
        b = np.broadcast_to(b, a.shape)
    return a, b


def besselj(n: Union[int, np.ndarray], z: Union[float, np.ndarray]) -> np.ndarray:
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


def spbessel(n: Union[int, np.ndarray], kr: Union[float, np.ndarray]) -> np.ndarray:
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


def sphankel2(n: Union[int, np.ndarray], kr: Union[float, np.ndarray]) -> np.ndarray:
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


def dspbessel(n: Union[int, np.ndarray], kr: Union[float, np.ndarray]) -> np.ndarray:
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


def dsphankel2(n: Union[int, np.ndarray], kr: Union[float, np.ndarray]) -> np.ndarray:
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
    kr: Union[float, np.ndarray],
    ka: Union[float, np.ndarray],
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
    return scale_factor * (
        spbessel(n, kr) - ((dspbessel(n, ka) / dsphankel2(n, ka)) * sphankel2(n, kr))
    )


def bn_open_omni(
    n: int, kr: Union[float, np.ndarray], normalize: bool = False
) -> np.ndarray:
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
        Radial filters of shape (N_mics, N_freqs, N_radial_filters).

    [1] Rafaely, B. (2015). Fundamentals of spherical array processing. Springer.
    """
    freqs = k * c / (2 * np.pi)
    omega = 2 * np.pi * freqs
    time_shift = np.exp(-1j * omega * delay)

    az_mic, elev_mic, r_mic = cart2sph(sphere_grid[0], sphere_grid[1], sphere_grid[2])

    ka = k * r_mic.mean(0)[None, None]  # scatter radius
    kr = k * r_mic.mean(0)[None, None]
    if np.any(kr[:, 0] == 0):
        kr[:, 0] = kr[:, 1]
    if np.any(ka[:, 0] == 0):
        ka[:, 0] = ka[:, 1]

    # NMLocatorSize = (order_max + 1) ** 2
    B_N = get_radial_filters_to_order_n(
        order_max, kr, ka, sphere_type, normalize=True
    ).squeeze(0)

    return B_N * time_shift[:, None]


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
    omega = np.linspace(0, 0.5, sig.shape[0] // 2 + 1)
    exp_omega = np.exp(-1j * 2 * np.pi * omega * delay_samples)
    exp_omega[-1] = np.real(exp_omega[-1])  # Fix Nyquist bin
    exp_omega = np.concatenate((exp_omega, np.conj(exp_omega[-2:0:-1])))

    # Apply complex delay in frequency domain
    Sig = np.fft.fft(sig, axis=0)
    Sig = Sig * exp_omega[:, np.newaxis]
    sig = np.fft.ifft(Sig, axis=0)

    return np.real(sig)


def hrir_group_delay(
    hrirs_l: np.ndarray, hrirs_r: np.ndarray, fs: int, NFFT: int
) -> Tuple[float, float]:
    """
    Estimate the group delay of a set of HRIRs.

    Parameters:
    hrirs (numpy.ndarray): Array of shape (N_samples, N_hrirs) containing the HRIRs.
    fs (int): Sampling frequency in Hz.
    NFFT (int): Number of FFT points to use for the group delay estimation.

    Returns:
    numpy.ndarray: Array of shape (N_hrirs,) containing the estimated group delays in seconds.
    """
    delays_l = []
    freqs = np.fft.rfftfreq(NFFT, d=1 / fs)
    for i in range(hrirs_l.shape[1]):
        # Calculate group delay for this specific direction
        # w=freqs ensures we look at relevant frequencies
        _, gd = group_delay((hrirs_l[:, i], 1), w=freqs, fs=fs)
        # Take the median across frequencies for this single direction
        delays_l.append(np.median(gd))

    # Now take the median across all directions to find the global shift
    grpDL = np.median(delays_l)

    # Repeat for Right Ear
    delays_r = []
    for i in range(hrirs_r.shape[1]):
        _, gd = group_delay((hrirs_r[:, i], 1), w=freqs, fs=fs)
        delays_r.append(np.median(gd))

    grpDR = np.median(delays_r)
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
            f"Warning: Discarding imaginary part with sum of [{imaginary_sum[0]:.2g}, {imaginary_sum[1]:.2g}] in rendering result."
        )
        binaural_output = np.real(binaural_output)

    return binaural_output


def rotate_sh(
    F_nm: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    sh_type: str = "real",
) -> np.ndarray:
    """Rotate spherical harmonics coefficients.
    See https://github.com/chris-hld/spaudiopy for the original implementation.

    Parameters
    ----------
    F_nm : (..., (N_sph+1)**2) numpy.ndarray
        Spherical harmonics coefficients
    yaw: float
        Rotation around Z axis.
    pitch: float
        Rotation around Y axis.
    roll: float
        Rotation around X axis.
    sh_type : 'complex' or 'real' spherical harmonics.
        Currently only 'real' is supported.

    Returns
    -------
    F_nm_rot : (..., (N_sph+1)**2) numpy.ndarray
        Rotated spherical harmonics coefficients.

    Examples
    --------
    .. plot::
        :context: close-figs

        N_sph = 5
        y1 = spa.sph.sh_matrix(N_sph, 0, np.pi/2, 'real')
        y2 = 0.5 * spa.sph.sh_matrix(N_sph, np.pi/3, np.pi/2, 'real')
        y_org = (4 * np.pi) / (N_sph + 1)**2 * (y1 + y2)
        y_rot = spa.sph.rotate_sh(y_org, -np.pi/2, np.pi/8, np.pi/4)

        spa.plot.sh_coeffs_subplot([y_org, y_rot],
                                   titles=['before', 'after'], cbar=False)

    """
    if sh_type == "complex":
        raise NotImplementedError("Currently only real valued SHs can be rotated")
    elif sh_type == "real":
        pass
    else:
        raise ValueError("Unknown SH type")

    N_sph = np.sqrt(F_nm.shape[-1]) - 1
    assert N_sph == np.floor(N_sph), "Invalid number of coefficients"
    N_sph = int(N_sph)

    Rot = sh_rotation_matrix(N_sph, yaw, pitch, roll, sh_type)

    return np.atleast_2d(F_nm) @ Rot.T



def sh_rotation_matrix(
    N_sph: int,
    yaw: float,
    pitch: float,
    roll: float,
    sh_type: str = "real",
    return_as_blocks: bool = False,
) -> Union[np.ndarray, list[np.ndarray]]:
    """Computes a Wigner-D matrix for the rotation of spherical harmonics.
    See https://github.com/chris-hld/spaudiopy for the original implementation.


    Parameters
    ----------
    N_sph : int
        Maximum SH order.
    yaw: float
        Rotation around Z axis.
    pitch: float
        Rotation around Y axis.
    roll: float
        Rotation around X axis.
    sh_type : 'complex' or 'real' spherical harmonics.
        Currently only 'real' is supported.
    return_as_blocks: optional, default is False.
        Return full block diagonal matrix, or a list of blocks if True.

    Returns
    -------
    R: (..., (N_sph+1)**2, (N_sph+1)**2) numpy.ndarray
        A block diagonal matrix R with shape (..., (N_sph+1)**2, (N_sph+1)**2),
        or a list r_blocks of numpy arrays [r(n) for n in range(N_sph)],
        where the shape of r is (..., 2*n-1, 2*n-1).

    See Also
    --------
    :py:func:`spaudiopy.sph.rotate_sh` : Apply rotation to SH signals.

    References
    ----------
    Implemented according to: Ivanic, Joseph, and Klaus Ruedenberg. "Rotation
    matrices for real spherical harmonics. Direct determination by recursion."
    The Journal of Physical Chemistry 100.15 (1996): 6342-6347.

    Ported from https://git.iem.at/audioplugins/IEMPluginSuite .

    """

    if sh_type == "complex":
        raise NotImplementedError("Currently only real valued SHs can be rotated")
    elif sh_type == "real":
        pass
    else:
        raise ValueError("Unknown SH type")

    rot_mat_cartesian = rotation_euler(yaw, pitch, roll)

    r_blocks = [np.array([[1]])]

    # change order to y, z, x
    r1 = rot_mat_cartesian[..., [1, 2, 0], :]
    r1 = r1[..., :, [1, 2, 0]]
    r_blocks.append(r1)

    # auxiliary functions
    def _rot_p_func(
        i: int, l: int, a: int, b: int, r1: np.ndarray, rlm1: np.ndarray
    ) -> np.ndarray:
        """Helper for recursion in Wigner-D rotation computation."""
        ri1 = r1[..., i + 1, 2]
        rim1 = r1[..., i + 1, 0]
        ri0 = r1[..., i + 1, 1]

        if b == -l:
            return (
                ri1 * rlm1[..., a + l - 1, 0] + rim1 * rlm1[..., a + l - 1, 2 * l - 2]
            )
        elif b == l:
            return (
                ri1 * rlm1[..., a + l - 1, 2 * l - 2] - rim1 * rlm1[..., a + l - 1, 0]
            )
        else:
            return ri0 * rlm1[..., a + l - 1, b + l - 1]

    def _rot_u_func(
        l: int, m: int, n: int, r1: np.ndarray, rlm1: np.ndarray
    ) -> np.ndarray:
        """U-term recursion helper."""
        return _rot_p_func(0, l, m, n, r1, rlm1)

    def _rot_v_func(
        l: int, m: int, n: int, r1: np.ndarray, rlm1: np.ndarray
    ) -> np.ndarray:
        """V-term recursion helper."""
        if m == 0:
            p0 = _rot_p_func(1, l, 1, n, r1, rlm1)
            p1 = _rot_p_func(-1, l, -1, n, r1, rlm1)
            return p0 + p1

        elif m > 0:
            p0 = _rot_p_func(1, l, m - 1, n, r1, rlm1)
            if m == 1:
                return p0 * np.sqrt(2)
            else:
                return p0 - _rot_p_func(-1, l, 1 - m, n, r1, rlm1)
        else:
            p1 = _rot_p_func(-1, l, -m - 1, n, r1, rlm1)
            if m == -1:
                return p1 * np.sqrt(2)
            else:
                return p1 + _rot_p_func(1, l, m + 1, n, r1, rlm1)

    def _rot_w_func(
        l: int, m: int, n: int, r1: np.ndarray, rlm1: np.ndarray
    ) -> np.ndarray:
        """W-term recursion helper."""
        if m > 0:
            p0 = _rot_p_func(1, l, m + 1, n, r1, rlm1)
            p1 = _rot_p_func(-1, l, -m - 1, n, r1, rlm1)
            return p0 + p1
        elif m < 0:
            p0 = _rot_p_func(1, l, m - 1, n, r1, rlm1)
            p1 = _rot_p_func(-1, l, 1 - m, n, r1, rlm1)
            return p0 - p1
        return 0

    rlm1 = r1
    for l in range(2, N_sph + 1):
        rl = np.zeros((2 * l + 1, 2 * l + 1))
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                d = int(m == 0)
                if abs(n) == l:
                    denom = (2 * l) * (2 * l - 1)
                else:
                    denom = l * l - n * n

                u = np.sqrt((l * l - m * m) / denom)
                v = (
                    np.sqrt((1.0 + d) * (l + abs(m) - 1.0) * (l + abs(m)) / denom)
                    * (1.0 - 2.0 * d)
                    * 0.5
                )
                w = (
                    np.sqrt((l - abs(m) - 1.0) * (l - abs(m)) / denom)
                    * (1.0 - d)
                    * (-0.5)
                )

                if u != 0:
                    u *= _rot_u_func(l, m, n, r1, rlm1)
                if v != 0:
                    v *= _rot_v_func(l, m, n, r1, rlm1)
                if w != 0:
                    w *= _rot_w_func(l, m, n, r1, rlm1)

                rl[..., m + l, n + l] = u + v + w

        r_blocks.append(rl)
        rlm1 = rl

    if return_as_blocks:
        return r_blocks
    else:
        # compose a block-diagonal matrix
        R = np.zeros(2 * [(N_sph + 1) ** 2])
        index = 0
        for r_block in r_blocks:
            R[
                ...,
                index : index + r_block.shape[-1],
                index : index + r_block.shape[-1],
            ] = r_block
            index += r_block.shape[-1]
        return R

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

def calculate_grid_weights(grid_cart):
    """
    Computes quadrature weights for an arbitrary grid on a sphere.
    grid_cart : (3, N) array of cartesian coordinates.
    Returns: (N,) array of weights summing to 4*pi.
    """
    # 1. Ensure points are on a unit sphere
    points = grid_cart.T
    points_unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Compute Voronoi regions
    # Note: Requires points to be unique and generally not all on one plane
    sv = SphericalVoronoi(points_unit)
    
    # 3. Calculate area of each Voronoi cell
    weights = sv.calculate_areas()
    
    return weights