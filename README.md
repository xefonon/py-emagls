# py-emagls
Unofficial Python implementation of End-to-End Magnitude Least Squares (eMagLS) binaural rendering for spherical microphone arrays. This project follows the method from the original paper and MATLAB reference implementation by Deppisch et al.

- Original MATLAB implementation: https://github.com/thomasdeppisch/eMagLS
- Paper: T. Deppisch, H. Helmholz, and J. Ahrens, "End-to-End Magnitude Least Squares Binaural Rendering of Spherical Microphone Array Signals," I3DA 2021. doi: 10.1109/I3DA48870.2021.9610864


## Requirements
- Python >= 3.12
- Dependencies listed in [pyproject.toml](pyproject.toml)

Install dependencies with your preferred tool (pip, uv, poetry). For example:
```bash
python -m pip install -e .
```

## Quick start
The main script demonstrates the full pipeline: load HRIRs, compute eMagLS filters, project microphone array signals to SH, and binaurally decode.

1) Place a SOFA HRIR file under signals/ (example in repo).
2) Provide a multichannel Eigenmike EM32 recording (32 channels) under signals/.
3) Run the script:

```bash
python main.py
```

Outputs:
- binaural_signal_test.wav
- binaural_rotated_signal_test.wav
- binaural_rotating_signal_test.wav

## Data expectations
- HRIRs are loaded from a SOFA file via `sofar` (see [Sofar](https://github.com/pyfar/sofar) package for details).
- The microphone array example assumes an Eigenmike EM32 geometry (32 capsules, 0.042 m radius).

## Notes and limitations
- This is not an official release from the original authors.
- Numerical results can differ from the MATLAB reference due to library differences and numerical precision.
- The example script uses a placeholder audio file name; you must provide a matching 32-channel recording.

## Citation
If you use this code in academic work, please cite the paper:

```bibtex
@inproceedings{Deppisch2021eMagLS,
	author = {Deppisch, Thomas and Helmholz, Heiko and Ahrens, Jens},
	title = {End-to-End Magnitude Least Squares Binaural Rendering of Spherical Microphone Array Signals},
	booktitle = {2021 Immersive and 3D Audio: from Architecture to Automotive (I3DA)},
	year = {2021},
	pages = {1--7},
	doi = {10.1109/I3DA48870.2021.9610864}
}
```

## License
See [LICENSE](LICENSE).
