# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
import soxr
from utils import cart2sph, get_em32_grid, sph2cart, binaural_decode, get_fade_window, sph_harm_all
from emagls import EndtoEnd_MagLS_weights
import sofar

if __name__ == "__main__":
# %%
    # Load HRIR data from .sofa file
    filename = os.path.join(".","signals", "ZTV406081722_1_processed.sofa")
    # /home/xekr/Repos/py-emagls/signals/MRT05.sofa
    with sofar.SofaStream(filename) as file:

        hrirs_left = file.Data_IR[:, 0, :].T  # Shape: (1625, 384)
        hrirs_right = file.Data_IR[:, 1, :].T  # Shape: (1625, 384)
        # get sampling frequency
        fs_hrir = file.Data_SamplingRate[:].item() # Shape: (1,)
        # get source positions (azimuth, elevation, radius)
        hrir_grid_sph = file.SourcePosition[:].T  # Shape: (1625, 3)
    
    # convert hrir_grid_sph[1] from elevation to zenith angle
    hrir_grid_sph[1] = 90 - hrir_grid_sph[1]
    hrir_grid_cart = sph2cart(hrir_grid_sph[0], hrir_grid_sph[1], hrir_grid_sph[2])  # Shape: (3, 1625)
    # plot hrir_grid in 3D to check if it looks correct (should be on a sphere)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(hrir_grid_cart[0, :], hrir_grid_cart[1, :], hrir_grid_cart[2, :], c="b", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    plt.title("HRIR Grid Positions")

    # %%
    # spherical microphone array positions
    mic_radius = np.array(0.042) # Radius of the microphone array in meters
    # Cartesian coordinates of the microphone array positions (shape: (3, 32))
    sph_mic_grid = get_em32_grid(mic_radius)
    az_mic, zenith_mic, _ = cart2sph(
        sph_mic_grid[0], sph_mic_grid[1], sph_mic_grid[2]
    )

    # Parameters
    c = 343 # Speed of sound in m/s
    order = 4 # Spherical harmonic order (4 gives 25 components, which is less than 32 microphones)
    NFFT = 1024 # Number of FFT points
    f_cut = 2000 # Cut-off frequency in Hz

    # plot Eigenmike em32 grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        sph_mic_grid[0, :], sph_mic_grid[1, :], sph_mic_grid[2, :], c="r", marker="o"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    plt.title("Eigenmike EM32 Microphone Positions")

    # Compute Magnitude Least Squares weights
    w_mls_l, w_mls_r = EndtoEnd_MagLS_weights(
        hrirs_left,
        hrirs_right,
        sph_mic_grid,
        hrir_grid_cart,
        fs_hrir,
        c,
        order,
        NFFT,
        f_cut,
        diffuseness_constraint=True,
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
    bin_sig = binaural_decode(PNM_mic, fs, w_mls_l, w_mls_r, fs)
    
    # Get the binaural signal with horizontal rotation (e.g., 90 degrees)
    bin_sig_rot = binaural_decode(
        PNM_mic, fs, w_mls_l, w_mls_r, fs, hor_rotation_angle_rad=np.pi / 2
    )

    # Save binaural signals
    sf.write("binaural_signal_test.wav", bin_sig, int(fs))
    sf.write("binaural_rotated_signal_test.wav", bin_sig_rot, int(fs))

    # Linspace of horizontal rotation angles
    num_rotations = 30
    hor_rot_angles = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)
    
    # Generate binaural signals for each horizontal rotation angle
    bin_sigs = []
    for angle in hor_rot_angles:
        bin_sig_rot = binaural_decode(
            PNM_mic, fs, w_mls_l, w_mls_r, fs, hor_rotation_angle_rad=angle
        )
        bin_sigs.append(bin_sig_rot)

    # Concatenate the rotated binaural signals along the time axis with fade in/out between them
    time_index = np.arange(0, len(audio), len(audio) // num_rotations)
    time_index = np.append(time_index, len(audio))
    
    # Collect each component (according to time index) and concatenate bin_sigs along the time axis
    fade_window = get_fade_window(
        len(audio) // num_rotations, 0.005, window_type="raised_cosine"
    )
    bin_sig_rot_all = np.concatenate(
        [
            fade_window[..., None] * bin_sig_rot[time_index[i] : time_index[i + 1]]
            for i, bin_sig_rot in enumerate(bin_sigs)
        ],
        axis=0,
    )
    # Save the concatenated rotated binaural signal
    sf.write("binaural_rotating_signal_test.wav", bin_sig_rot_all, int(fs))

# %%
