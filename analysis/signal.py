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
                      f_sampling_hz):
    """
    Isolate FW and BW pulses by zeroing everything outside each gate,
    then compute xcorr_cpi time delay.
    Returns time_delay in seconds, or NaN if either gate is all zeros.
    """
    fw_isolated = np.zeros_like(signal)
    bw_isolated = np.zeros_like(signal)
    fw_isolated[fw_gate_idx0:fw_gate_idx1] = signal[fw_gate_idx0:fw_gate_idx1]
    bw_isolated[bw_gate_idx0:bw_gate_idx1] = signal[bw_gate_idx0:bw_gate_idx1]

    return xcorr_cpi(fw_isolated, bw_isolated, f_sampling_hz)