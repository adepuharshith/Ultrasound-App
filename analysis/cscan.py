import numpy as np
from scipy.signal import hilbert


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


def compute_wave_speed_map(tof_map, thickness_m):
    """
    Wave speed from ToF map and known sample thickness.
    tof_map: in µs
    thickness_m: in metres
    Returns wave speed map in m/s.
    """
    tof_s = tof_map * 1e-6          # convert µs → s
    # round trip: wave travels 2x thickness
    return (2 * thickness_m) / tof_s


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