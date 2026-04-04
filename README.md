# Ultrasound NDE Analysis Tool

A Streamlit web application for interactive analysis of ultrasonic nondestructive evaluation (NDE) scan data. Built for processing 2D raster C-scans acquired from immersion or contact ultrasound systems.

## What it does

The app provides two analysis modes:

### Mode 1 — C-scan (2D Raster)

Upload a pair of files:
- **`.txt` metadata file** — contains acquisition parameters and scan geometry
- **`.dat` binary data file** — raw waveform data as 16-bit integers

The app parses the metadata, loads the binary scan into a 3D array `(index_points × scan_points × samples_per_waveform)`, and provides:

- **Waveform viewer** — displays the RF waveform at any selected scan position. Click a pixel on the C-scan to jump to that waveform.
- **Gate controls** — configure up to three time gates:
  - **FW gate** — Front wall echo window
  - **BW gate** — Back wall echo window
  - **Data gate** — Custom analysis window
- **Dual C-scan maps** — two independent colormaps shown side by side, each computing one of:
  - **Peak amplitude** — maximum envelope amplitude within a gate
  - **Time of Flight (ToF)** — time at peak envelope amplitude within a gate (µs)
  - **Wave speed (xcorr)** — longitudinal wave speed (m/s) computed via cross-correlation of the front wall and back wall pulses, using user-provided sample thickness
- **Acquisition info panel** — displays transducer frequency, voltage, damping, gain, scan geometry, and sampling parameters parsed from the file

### Mode 2 — Single Waveform (CSV)

Upload a single-column `.csv` file containing one waveform. The app plots amplitude vs. sample index. Gate-based ToF measurement is included as a placeholder for future development.

## File naming convention

The `.txt` metadata filename must follow this underscore-delimited convention:

```
[sample]_[MHz]_[startTime_us]_[windowLength_us]_[V]_[pF]_[Ohm]_[dB]_[anything...].txt
```

Example: `Sample1_5MHz_0us_20us_100V_50pF_50Ohm_20dB_scan.txt`

Acquisition parameters are parsed directly from the filename segments.

## Project structure

```
Ultrasound-App/
├── app.py                  # Streamlit UI and main application logic
├── requirements.txt        # Python dependencies
├── analysis/
│   ├── io.py               # File I/O: metadata parsing and data loading
│   ├── cscan.py            # C-scan computations (amplitude, ToF, wave speed maps)
│   └── signal.py           # Signal processing: cross-correlation, Tukey windowing
```

### Module overview

**`analysis/io.py`**
- `parse_metadata_from_txt` — reads acquisition parameters from the `.txt` file (supports UTF-16 and UTF-8 encodings) and derives the time axis
- `load_cscan_data` — loads the binary `.dat` file, normalises 16-bit integers to `[-1, 1]` range, and reshapes into `(index, scan, samples)`
- `load_single_waveform_csv` — loads a single waveform from CSV

**`analysis/cscan.py`**
- `compute_peak_amplitude_map` — max Hilbert envelope amplitude within a gated window
- `compute_tof_map` — time at peak envelope within a gated window
- `compute_wave_speed_xcorr` — per-pixel wave speed from FW/BW pulse cross-correlation and known sample thickness
- `build_spatial_axes` — converts scan/index point counts and increments to mm axes
- `time_to_index` — converts a time in µs to the nearest sample index

**`analysis/signal.py`**
- `xcorr_cpi` — cross-correlation time delay with sub-sample accuracy via quadratic peak fitting
- `isolate_and_xcorr` — isolates FW and BW pulses using gate masks, then calls `xcorr_cpi`
- `apply_tukey_window` / `extract_windowed_signal` — Tukey windowing utilities

## Installation and running

```bash
# Clone the repo
git clone https://github.com/adepuharshith/Ultrasound-App.git
cd Ultrasound-App

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Dependencies

Key packages: `streamlit`, `numpy`, `scipy`, `plotly`, `pandas`. See `requirements.txt` for the full pinned list.
