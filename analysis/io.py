import numpy as np
import pandas as pd
import io

def parse_metadata_from_txt(txt_file):
    raw_bytes = txt_file.read()
    if raw_bytes[:2] == b'\xff\xfe':
        content = raw_bytes[2:].decode('utf-16-le')
    else:
        try:
            content = raw_bytes.decode('utf-16-le')
        except:
            content = raw_bytes.decode('utf-8', errors='replace')

    file_name      = txt_file.name
    file_name_data = file_name.replace('.dat.txt', '').replace('.txt', '')
    parts          = file_name.split('_')

    # ── Fixed positional parsing: parts[1] through parts[7] ────────
    # Convention: [sample]_[MHz]_[us]_[us]_[V]_[pF]_[Ohm]_[dB]_[anything...]
    try:
        metadata = {
            'file_name'        : file_name,
            'file_name_data'   : file_name_data,
            'transducer'       : float(parts[1].replace('p', '.').replace('MHz', '')),
            'start_time'       : float(parts[2].replace('us', '')),
            'length_of_signal' : float(parts[3].replace('us', '')),
            'voltage'          : float(parts[4].replace('V', '')),
            'capacitance'      : float(parts[5].replace('pF', '')),
            'damping'          : float(parts[6].replace('Ohm', '')),
            'gain'             : float(parts[7].replace('dB', '')),
        }
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Filename does not follow the expected convention:\n"
            f"[sample]_[MHz]_[us]_[us]_[V]_[pF]_[Ohm]_[dB]_[anything...]\n"
            f"Got: {file_name}\nError: {e}"
        )

    # ── Parse file content ──────────────────────────────────────────
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if 'Scan points' in line:
            metadata['scan_points']       = int(line.split(':')[1].strip())
        elif 'Index points' in line:
            metadata['index_points']      = int(line.split(':')[1].strip())
        elif 'Scan increment' in line:
            metadata['scan_increment']    = float(line.split(':')[1].strip().split()[0])
        elif 'Index increment' in line:
            metadata['index_increment']   = float(line.split(':')[1].strip().split()[0])
        elif 'Samples per waveform' in line:
            metadata['number_of_samples'] = int(line.split(':')[1].strip())
        elif 'Sampling frequency' in line:
            metadata['f_sampling']        = float(line.split(':')[1].strip().split()[0])
        elif 'Data type' in line:
            metadata['data_type']         = int(line.split(':')[1].strip().split()[0])

    # ── Derived quantities ──────────────────────────────────────────
    metadata['f_sampling_hz']   = metadata['f_sampling'] * 1e6
    metadata['sample_interval'] = 1 / metadata['f_sampling_hz'] * 1e6
    metadata['time_axis']       = (
        metadata['start_time'] +
        np.arange(metadata['number_of_samples']) * metadata['sample_interval']
    )

    return metadata


def load_cscan_data(dat_file, metadata):
    """
    Load the binary .dat file into a normalized 3D numpy array.
    dat_file: a Streamlit UploadedFile object
    metadata: dict returned by parse_metadata_from_txt
    Returns: ndarray of shape (index_points, scan_points, samples_per_waveform)
    """
    scan_points          = metadata['scan_points']
    index_points         = metadata['index_points']
    samples_per_waveform = metadata['number_of_samples']
    total_samples        = scan_points * index_points * samples_per_waveform

    raw_bytes = dat_file.read()
    data      = np.frombuffer(raw_bytes, dtype=np.int16, count=total_samples)
    data_norm = data / 2048.0

    return data_norm.reshape((index_points, scan_points, samples_per_waveform))


def get_center_waveform(waveform_data):
    """
    Extract the waveform at the center of the scan grid.
    Returns: 1D numpy array
    """
    ci = waveform_data.shape[0] // 2
    cj = waveform_data.shape[1] // 2
    return waveform_data[ci, cj, :]


def load_single_waveform_csv(csv_file):
    """
    Load a single waveform from a CSV file.
    Expects one column of amplitude values.
    csv_file: a Streamlit UploadedFile object
    Returns: 1D numpy array
    """
    df = pd.read_csv(csv_file, header=None)
    return df.iloc[:, 0].to_numpy(dtype=np.float64)