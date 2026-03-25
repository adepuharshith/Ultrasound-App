import streamlit as st
import numpy as np
import plotly.graph_objects as go
from analysis.io import (
    parse_metadata_from_txt,
    load_cscan_data,
    get_center_waveform,
    load_single_waveform_csv,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Ultrasound NDE", layout="wide")
st.title("Ultrasound NDE Analysis Tool")

# ─────────────────────────────────────────────
# MODE SELECTION
# ─────────────────────────────────────────────
mode = st.radio(
    "Select mode",
    ["C-scan (2D raster)", "Single waveform (CSV)"],
    horizontal=True
)

st.divider()

# ═════════════════════════════════════════════
# MODE 1 — C-SCAN
# ═════════════════════════════════════════════
if mode == "C-scan (2D raster)":

    # ── File upload ──────────────────────────
    col_upload_left, col_upload_right = st.columns([2, 2])
    with col_upload_left:
        txt_file = st.file_uploader("Upload metadata file (.txt)", type=["txt"])
    with col_upload_right:
        dat_file = st.file_uploader("Upload data file (.dat)", type=["dat"])

    if txt_file and dat_file:

        # ── Parse metadata ───────────────────
        try:
            metadata = parse_metadata_from_txt(txt_file)
        except Exception as e:
            st.error(f"Failed to parse metadata: {e}")
            st.stop()

        # ── Load data ────────────────────────
        with st.spinner("Loading scan data..."):
            try:
                waveform_data = load_cscan_data(dat_file, metadata)
            except Exception as e:
                st.error(f"Failed to load .dat file: {e}")
                st.stop()

        time_axis = metadata['time_axis']

        # ── Layout: sidebar info | waveform | gate controls ──
        col_info, col_wave, col_gates = st.columns([1, 3, 1])

        # ── Acquisition parameters ───────────
        with col_info:
            st.subheader("Acquisition")
            st.markdown(f"**Transducer:** {metadata['transducer']} MHz")
            st.markdown(f"**Voltage:** {metadata['voltage']} V")
            st.markdown(f"**Damping:** {metadata['damping']} Ω")
            st.markdown(f"**Gain:** {metadata['gain']} dB")
            st.markdown(f"**LPF/HPF:** — ")
            st.divider()
            st.subheader("Scan geometry")
            st.markdown(f"**Scan points:** {metadata['scan_points']}")
            st.markdown(f"**Index points:** {metadata['index_points']}")
            st.markdown(f"**Scan increment:** {metadata['scan_increment']} mm")
            st.markdown(f"**Index increment:** {metadata['index_increment']} mm")
            st.divider()
            st.subheader("Sampling")
            st.markdown(f"**Fs:** {metadata['f_sampling']} MHz")
            st.markdown(f"**Samples/wfm:** {metadata['number_of_samples']}")
            st.markdown(f"**Start time:** {metadata['start_time']} µs")
            st.markdown(f"**Window:** {metadata['length_of_signal']} µs")

        # ── Gate controls ────────────────────
        with col_gates:
            st.subheader("Gates")

            fw_enabled = st.checkbox("FW gate", value=False)
            if fw_enabled:
                fw_start = st.number_input(
                    "FW start (µs)", value=float(round(time_axis[0], 2)),
                    step=0.1, format="%.2f", key="fw_start"
                )
                fw_end = st.number_input(
                    "FW end (µs)", value=float(round(time_axis[0] + 2.0, 2)),
                    step=0.1, format="%.2f", key="fw_end"
                )

            st.divider()
            bw_enabled = st.checkbox("BW gate", value=False)
            if bw_enabled:
                bw_start = st.number_input(
                    "BW start (µs)", value=float(round(time_axis[-1] - 4.0, 2)),
                    step=0.1, format="%.2f", key="bw_start"
                )
                bw_end = st.number_input(
                    "BW end (µs)", value=float(round(time_axis[-1] - 1.0, 2)),
                    step=0.1, format="%.2f", key="bw_end"
                )

            st.divider()
            data_gate_enabled = st.checkbox("Data gate", value=False)
            if data_gate_enabled:
                dg_start = st.number_input(
                    "Data gate start (µs)", value=float(round(time_axis[0], 2)),
                    step=0.1, format="%.2f", key="dg_start"
                )
                dg_end = st.number_input(
                    "Data gate end (µs)", value=float(round(time_axis[-1], 2)),
                    step=0.1, format="%.2f", key="dg_end"
                )

        # ── Center waveform plot ─────────────
        with col_wave:
            st.subheader("Center waveform")
            center_wf = get_center_waveform(waveform_data)

            fig_wf = go.Figure()

            # Raw waveform
            fig_wf.add_trace(go.Scatter(
                x=time_axis, y=center_wf,
                mode='lines',
                line=dict(color='steelblue', width=1),
                name='Waveform'
            ))

            # FW gate shading
            if fw_enabled:
                fig_wf.add_vrect(
                    x0=fw_start, x1=fw_end,
                    fillcolor="rgba(255,160,122,0.25)",
                    line_width=1, line_color="tomato",
                    annotation_text="FW", annotation_position="top left"
                )

            # BW gate shading
            if bw_enabled:
                fig_wf.add_vrect(
                    x0=bw_start, x1=bw_end,
                    fillcolor="rgba(144,238,144,0.25)",
                    line_width=1, line_color="seagreen",
                    annotation_text="BW", annotation_position="top left"
                )

            # Data gate shading
            if data_gate_enabled:
                fig_wf.add_vrect(
                    x0=dg_start, x1=dg_end,
                    fillcolor="rgba(180,180,180,0.2)",
                    line_width=1, line_color="gray",
                    annotation_text="Data", annotation_position="top left"
                )

            fig_wf.update_layout(
                xaxis_title="Time (µs)",
                yaxis_title="Amplitude (normalized)",
                height=350,
                margin=dict(l=20, r=20, t=30, b=40),
                legend=dict(orientation="h", y=1.1),
                hovermode="x unified"
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    else:
        st.info("Upload both the .txt and .dat files to begin.")


# ═════════════════════════════════════════════
# MODE 2 — SINGLE WAVEFORM CSV
# ═════════════════════════════════════════════
else:
    csv_file = st.file_uploader("Upload single waveform (.csv)", type=["csv"])

    if csv_file:
        try:
            waveform = load_single_waveform_csv(csv_file)
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.stop()

        sample_index = np.arange(len(waveform))

        col_plot, col_result = st.columns([3, 1])

        with col_plot:
            st.subheader("Waveform")

            fig_csv = go.Figure()
            fig_csv.add_trace(go.Scatter(
                x=sample_index, y=waveform,
                mode='lines',
                line=dict(color='steelblue', width=1),
                name='Waveform'
            ))
            fig_csv.update_layout(
                xaxis_title="Sample index",
                yaxis_title="Amplitude",
                height=350,
                margin=dict(l=20, r=20, t=30, b=40),
                hovermode="x unified"
            )
            st.plotly_chart(fig_csv, use_container_width=True)

            st.caption("Note: x-axis is in sample index — enter sampling frequency below to convert to µs.")

        with col_result:
            st.subheader("ToF measurement")
            fs_input = st.number_input(
                "Sampling frequency (MHz)", value=125.0, step=1.0
            )
            st.info("Gate-based ToF measurement will be added in the next step.")

    else:
        st.info("Upload a single waveform CSV file to begin.")