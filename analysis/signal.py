import numpy as np
from scipy.signal import correlate, hilbert, savgol_filter
from scipy.signal.windows import tukey
from scipy.interpolate import CubicSpline


def calculate_correlation(pulse_1, pulse_2):
    correlation     = correlate(pulse_2, pulse_1, mode='full')
    lags            = np.arange(-len(pulse_1) + 1, len(pulse_2))
    max_corr_index  = np.argmax(correlation)
    return correlation, lags, max_corr_index


def quadratic_fit(lags, correlation, max_corr_index, fit_range=2):
    x_fit        = lags[max_corr_index - fit_range : max_corr_index + fit_range + 1]
    y_fit        = correlation[max_corr_index - fit_range : max_corr_index + fit_range + 1]
    coefficients = np.polyfit(x_fit, y_fit, 2)
    a, b, _      = coefficients
    vertex       = -b / (2 * a)
    return vertex


def xcorr_cpi(pulse_1, pulse_2, f_sampling_hz):
    """
    Cross-correlation time delay with sub-sample accuracy via quadratic fit.
    Returns time delay in seconds, or NaN if either pulse is all zeros.
    """
    if np.all(pulse_1 == 0) or np.all(pulse_2 == 0):
        return np.nan
    correlation, lags, max_corr_index = calculate_correlation(pulse_1, pulse_2)
    if max_corr_index < 2 or max_corr_index + 2 >= len(correlation):
        return np.nan
    vertex               = quadratic_fit(lags, correlation, max_corr_index)
    time_delay_seconds   = vertex / f_sampling_hz
    return time_delay_seconds


def apply_tukey_window(signal_segment, alpha=0.5):
    window = tukey(len(signal_segment), alpha=alpha)
    return signal_segment * window


def extract_windowed_signal(signal, center_index, pulse_width_samples, window_alpha=0.5):
    """
    Extract a Tukey-windowed segment centred at center_index.
    Returns (windowed_segment, start_idx, end_idx).
    """
    start            = max(center_index - int(pulse_width_samples / 2), 0)
    end              = min(center_index + int(pulse_width_samples / 2), len(signal))
    windowed_signal  = apply_tukey_window(signal[start:end], alpha=window_alpha)
    return windowed_signal, start, end


def isolate_and_xcorr(signal, fw_gate_idx0, fw_gate_idx1,
                      bw_gate_idx0, bw_gate_idx1,
                      pulse_width_samples, f_sampling_hz,
                      fw_amp_threshold=0.3):
    """
    For a single waveform:
      1. Find max envelope peak within FW gate → extract Tukey-windowed pulse
      2. Find max envelope peak within BW gate → extract Tukey-windowed pulse
      3. xcorr_cpi on the two pulses → time delay in seconds

    Returns:
        fw_amp       : float  — peak amplitude of FW pulse (for thresholding)
        bw_amp       : float  — peak amplitude of BW pulse
        fw_tof_us    : float  — time of FW peak in µs (sample_interval already in caller)
        bw_tof_us    : float  — time of BW peak in µs
        time_delay_s : float  — xcorr time delay in seconds (NaN if below threshold)
    """
    # Smooth + envelope for robust peak finding
    smooth     = savgol_filter(signal, window_length=15, polyorder=2)
    envelope   = np.abs(hilbert(smooth))

    # ── FW peak within gate ───────────────────
    fw_env_segment  = envelope[fw_gate_idx0:fw_gate_idx1]
    fw_peak_rel     = int(np.argmax(fw_env_segment))
    fw_peak_abs     = fw_gate_idx0 + fw_peak_rel
    fw_amp          = envelope[fw_peak_abs]

    if fw_amp < fw_amp_threshold:
        return fw_amp, np.nan, np.nan, np.nan, np.nan

    fw_windowed, fw_s, fw_e = extract_windowed_signal(signal, fw_peak_abs, pulse_width_samples)

    # ── BW peak within gate ───────────────────
    bw_env_segment  = envelope[bw_gate_idx0:bw_gate_idx1]
    bw_peak_rel     = int(np.argmax(bw_env_segment))
    bw_peak_abs     = bw_gate_idx0 + bw_peak_rel
    bw_amp          = envelope[bw_peak_abs]

    bw_windowed, bw_s, bw_e = extract_windowed_signal(signal, bw_peak_abs, pulse_width_samples)

    # ── Build full-length isolated signals ────
    fw_isolated          = np.zeros_like(signal)
    bw_isolated          = np.zeros_like(signal)
    fw_isolated[fw_s:fw_e] = fw_windowed
    bw_isolated[bw_s:bw_e] = bw_windowed

    # ── xcorr ────────────────────────────────
    try:
        time_delay_s = xcorr_cpi(fw_isolated, bw_isolated, f_sampling_hz)
    except Exception:
        time_delay_s = np.nan

    # ToF of peak sample in µs — caller passes sample_interval separately
    fw_tof_sample = fw_peak_abs
    bw_tof_sample = bw_peak_abs

    return fw_amp, bw_amp, fw_tof_sample, bw_tof_sample, time_delay_s