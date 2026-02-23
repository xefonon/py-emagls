# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
import soxr
from utils import (
    cart2sph,
    get_em32_grid,
    sph2cart,
    binaural_decode,
    sph_harm_all,
    fibonacci_grid_sphere,
    plot_diffuse_field_validation,
    calculate_smr,
    plot_smr_err_freq_vs_azimuth,
)
from emagls import EndtoEnd_MagLS_weights, EndtoEnd_LS_weights
import sofar
from pathlib import Path

if __name__ == "__main__":
    # %%
    # Load HRIR data from .sofa file
    filename = Path("signals") / "ZTV406081722_1_processed.sofa"
    # /home/xekr/Repos/py-emagls/signals/MRT05.sofa
    with sofar.SofaStream(filename) as file:
        hrirs_left = file.Data_IR[:, 0, :].T  # Shape: (1625, 384)
        hrirs_right = file.Data_IR[:, 1, :].T  # Shape: (1625, 384)
        # get sampling frequency
        fs_hrir = file.Data_SamplingRate[:].item()  # Shape: (1,)
        # get source positions (azimuth, elevation, radius)
        hrir_grid_sph = file.SourcePosition[:].T  # Shape: (1625, 3)

    # convert hrir_grid_sph[1] from elevation to zenith angle
    hrir_grid_sph[1] = 90 - hrir_grid_sph[1]
    hrir_grid_cart = sph2cart(
        hrir_grid_sph[0], hrir_grid_sph[1], hrir_grid_sph[2]
    )  # Shape: (3, 1625)
    # plot hrir_grid in 3D to check if it looks correct (should be on a sphere)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        hrir_grid_cart[0, :],
        hrir_grid_cart[1, :],
        hrir_grid_cart[2, :],
        c="b",
        marker="o",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    plt.title("HRIR Grid Positions")

    # %%
    # spherical microphone array positions
    mic_radius = np.array(0.042)  # Radius of the microphone array in meters
    Nref = 20
    # Cartesian coordinates of the microphone array positions (shape: (3, 32))
    sph_mic_grid = get_em32_grid(mic_radius)
    az_mic, zenith_mic, _ = cart2sph(sph_mic_grid[0], sph_mic_grid[1], sph_mic_grid[2])

    # Parameters
    c = 343  # Speed of sound in m/s
    order = 4  # Spherical harmonic order (4 gives 25 components, which is less than 32 microphones)
    NFFT = 2048  # Number of FFT points
    f_cut = 1500  # Cut-off frequency in Hz

    # plot Eigenmike em32 grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # project over a solid sphere for visualization
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    sph_radius = mic_radius*0.99
    x = sph_radius * np.cos(u) * np.sin(v)
    y = sph_radius * np.sin(u) * np.sin(v)
    z = sph_radius * np.cos(v)
    ax.plot_surface(x, y, z, color="c", alpha=0.2)
    ax.scatter(
        sph_mic_grid[0, :], sph_mic_grid[1, :], sph_mic_grid[2, :], c="r", marker="o"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    plt.title("Eigenmike EM32 Microphone Positions")
    plt.savefig("em32_grid.png", dpi=300)

    # Compute Magnitude Least Squares weights
    w_mls_l_em32, w_mls_r_em32 = EndtoEnd_MagLS_weights(
        hrirs_left,
        hrirs_right,
        sph_mic_grid,
        hrir_grid_cart,
        fs_hrir,
        c,
        order,
        NFFT,
        f_cut,
        diffuseness_constraint=False,
    )
    # Compute Least Squares weights for eigenmike em32 grid (without magnitude constraint)
    w_ls_l_em32, w_ls_r_em32 = EndtoEnd_LS_weights(
        hrirs_left,
        hrirs_right,
        sph_mic_grid,
        hrir_grid_cart,
        fs_hrir,
        c,
        order,
        NFFT,
        f_cut,
    )

    high_res_grid = fibonacci_grid_sphere(
        (Nref + 1) ** 2
    )  # High-resolution grid with 625 points
    # Compute Magnitude Least Squares weights for high-resolution grid
    w_ref_l, w_ref_r = EndtoEnd_LS_weights(
        hrirs_left, hrirs_right, high_res_grid, hrir_grid_cart, fs_hrir, c, Nref, NFFT
    )

    plot_diffuse_field_validation(
        w_mls_l_em32,
        w_mls_r_em32,
        w_ls_l_em32,
        w_ls_r_em32,
        hrirs_left=hrirs_left,
        hrirs_right=hrirs_right,
        hrir_grid_cart=hrir_grid_cart,
        order=order,
        fs=fs_hrir,
        title="EmagLS vs LS",
    )
    # Calculate the Summed Magnitude Response (SMR) and error compared to the high-resolution reference
    freqs, smr_mls_db, smr_ref_db, smr_err_db = calculate_smr(
        w_mls_l_em32,
        w_mls_r_em32,
        w_ref_l,
        w_ref_r,
        fs=fs_hrir,
        order_mls=order,
        order_ref=Nref,  # Use the high-resolution grid weights as reference
        NFFT=NFFT,
    )
    # Plot the SMR error vs. azimuth for the horizontal plane
    grid_cart = fibonacci_grid_sphere(2500)  # must match K used in calculate_smr

    f_cut = 16000
    f_index_cut = np.where(freqs <= f_cut)[0][-1] + 1
    freqs = freqs[:f_index_cut]
    smr_err_db = smr_err_db[:f_index_cut, :]
    plot_smr_err_freq_vs_azimuth(
        freqs,
        smr_err_db,
        grid_cart,
        n_az_bins=120,
        horiz_only=True,
        title="Summed Magnitude Response Error (dB) vs. Azimuth",
        clim=(-12, 12),
    )

    # Microphone array spherical harmonic basis functions
    Y_NM_mic = sph_harm_all(order, az_mic, zenith_mic, kind="real")

    # read the audio file
    EM32_audio_path = os.path.join("signals", "solo_guitar.wav")
    audio, fs = sf.read(EM32_audio_path)

    # resample if necessary
    if fs != fs_hrir:
        resampled_audio_channels = []
        for channel in range(audio.shape[1]):
            resampled_channel = soxr.resample(audio[:, channel], fs, fs_hrir)
            resampled_audio_channels.append(resampled_channel)
        audio = np.stack(resampled_audio_channels, axis=-1)
        fs = fs_hrir

    # Trim or pad audio to a fixed length (e.g., 30 seconds) for testing
    time_to_play = 30  # seconds
    time_to_start = 3  # seconds
    audio = audio[
        int(fs * time_to_start) : int(fs * time_to_start) + int(fs * time_to_play)
    ]

    # Project the audio onto the spherical harmonic basis to get the SH-domain (Ambisonic) representation
    PNM_mic = np.einsum("nm,mt -> tn", np.linalg.pinv(Y_NM_mic), audio.T)

    # Decode the SH-domain signal to binaural using the computed Magnitude Least Squares weights
    bin_sig = binaural_decode(PNM_mic, fs, w_mls_l_em32, w_mls_r_em32, fs)

    # Get the binaural signal with horizontal rotation (e.g., 90 degrees)
    bin_sig_rot = binaural_decode(
        PNM_mic, fs, w_mls_l_em32, w_mls_r_em32, fs, hor_rotation_angle_rad=np.pi / 2
    )

    # Save binaural signals
    sf.write("binaural_signal_test.wav", bin_sig, int(fs))
    sf.write("binaural_rotated_signal_test.wav", bin_sig_rot, int(fs))


# %%
