import numpy as np
from scipy.signal import hilbert
from analysis.signal import isolate_and_xcorr

def compute_envelope(waveform):
    """Analytic envelope of a 1D waveform via Hilbert transform."""
    return np.abs(hilbert(waveform))


def compute_amplitude_map(waveform_data):
    """
    Default map — max envelope amplitude across the full time window.
    No gate required.
    shape: (index_points, scan_points)
    """
    envelope = np.abs(hilbert(waveform_data, axis=2))
    return envelope.max(axis=2)


def compute_peak_amplitude_map(waveform_data, gate_start_idx, gate_end_idx):
    """
    Peak envelope amplitude within a gate window.
    """
    gated    = waveform_data[:, :, gate_start_idx:gate_end_idx]
    envelope = np.abs(hilbert(gated, axis=2))
    return envelope.max(axis=2)


def compute_tof_map(waveform_data, gate_start_idx, gate_end_idx, time_axis):
    """
    Time of peak envelope amplitude within a gate window.
    Returns map in µs.
    """
    gated      = waveform_data[:, :, gate_start_idx:gate_end_idx]
    envelope   = np.abs(hilbert(gated, axis=2))
    peak_idx   = envelope.argmax(axis=2)
    return time_axis[gate_start_idx:gate_end_idx][peak_idx]


def time_to_index(time_us, time_axis):
    """Convert a time in µs to the nearest sample index."""
    return int(np.argmin(np.abs(time_axis - time_us)))


def build_spatial_axes(metadata):
    """
    Build physical spatial axes in mm from scan geometry.
    Returns x_axis (scan direction), y_axis (index direction).
    """
    x_axis = np.arange(metadata['scan_points'])  * metadata['scan_increment']
    y_axis = np.arange(metadata['index_points']) * metadata['index_increment']
    return x_axis, y_axis

def compute_wave_speed_xcorr(waveform_data, fw_gate_us, bw_gate_us,
                              pulse_width_us, time_axis,
                              f_sampling_hz, sample_interval_us,
                              thickness_m, fw_amp_threshold=0.3):
    """
    Compute wave speed map using FW/BW pulse isolation + xcorr_cpi.

    Parameters:
        waveform_data     : (index, scan, samples) normalised float array
        fw_gate_us        : (start_us, end_us) tuple — FW search window
        bw_gate_us        : (start_us, end_us) tuple — BW search window
        pulse_width_us    : float — Tukey window width in µs
        time_axis         : 1D array of time values in µs
        f_sampling_hz     : float — sampling frequency in Hz
        sample_interval_us: float — 1/f_sampling in µs
        thickness_m       : float — sample thickness in metres
        fw_amp_threshold  : float — skip pixel if FW peak below this (default 0.3)

    Returns:
        wave_speed_map    : (index, scan) array in m/s, NaN where rejected
        tof_map_us        : (index, scan) array of xcorr time delays in µs
    """
    n_index, n_scan, _ = waveform_data.shape
    pulse_width_samples = int(pulse_width_us / sample_interval_us)

    fw_idx0 = time_to_index(fw_gate_us[0], time_axis)
    fw_idx1 = time_to_index(fw_gate_us[1], time_axis)
    bw_idx0 = time_to_index(bw_gate_us[0], time_axis)
    bw_idx1 = time_to_index(bw_gate_us[1], time_axis)

    wave_speed_map = np.full((n_index, n_scan), np.nan)
    tof_map_us     = np.full((n_index, n_scan), np.nan)

    for i in range(n_index):
        for j in range(n_scan):
            signal = waveform_data[i, j, :]
            _, _, _, _, time_delay_s = isolate_and_xcorr(
                signal, fw_idx0, fw_idx1, bw_idx0, bw_idx1,
                pulse_width_samples, f_sampling_hz, fw_amp_threshold
            )
            if not np.isnan(time_delay_s) and time_delay_s > 0:
                wave_speed_map[i, j] = 2.0 * thickness_m / time_delay_s
                tof_map_us[i, j]     = time_delay_s * 1e6

    return wave_speed_map, tof_map_us