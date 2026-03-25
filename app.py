import streamlit as st
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from analysis.io import (
    parse_metadata_from_txt,
    load_cscan_data,
    get_center_waveform,
    load_single_waveform_csv,
)
from analysis.cscan import (
    compute_amplitude_map,
    compute_peak_amplitude_map,
    compute_tof_map,
    compute_wave_speed_map,
    compute_envelope,
    time_to_index,
    build_spatial_axes,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Ultrasound NDE", layout="wide")
st.title("Ultrasound NDE Analysis Tool")

mode = st.radio("Select mode", ["C-scan (2D raster)", "Single waveform (CSV)"], horizontal=True)
st.divider()

# ═════════════════════════════════════════════
# MODE 1 — C-SCAN
# ═════════════════════════════════════════════
if mode == "C-scan (2D raster)":

    # ── File upload row ───────────────────────
    col_upload_left, col_upload_right = st.columns([2, 2])
    with col_upload_left:
        txt_file = st.file_uploader("Upload metadata file (.txt)", type=["txt"])
    with col_upload_right:
        dat_file = st.file_uploader("Upload data file (.dat)", type=["dat"])

    st.divider()

    # ── Always-visible layout ─────────────────
    col_info, col_wave, col_gates = st.columns([1, 3, 1])

    # ── Acquisition panel — always visible ───
    with col_info:
        st.subheader("Acquisition")
        if txt_file:
            try:
                metadata = parse_metadata_from_txt(txt_file)
                parsed_ok = True
            except Exception as e:
                st.error(f"Metadata error: {e}")
                parsed_ok = False
                metadata  = {}
        else:
            parsed_ok = False
            metadata  = {}

        def show_val(label, key, unit=""):
            val = metadata.get(key, "—")
            st.markdown(f"**{label}:** {val}{(' ' + unit) if val != '—' else ''}")

        show_val("Transducer", "transducer", "MHz")
        show_val("Voltage",    "voltage",    "V")
        show_val("Damping",    "damping",    "Ω")
        show_val("Gain",       "gain",       "dB")
        st.divider()
        st.subheader("Scan geometry")
        show_val("Scan points",     "scan_points")
        show_val("Index points",    "index_points")
        show_val("Scan increment",  "scan_increment",  "mm")
        show_val("Index increment", "index_increment", "mm")
        st.divider()
        st.subheader("Sampling")
        show_val("Fs",           "f_sampling",      "MHz")
        show_val("Samples/wfm",  "number_of_samples")
        show_val("Start time",   "start_time",      "µs")
        show_val("Window",       "length_of_signal","µs")

    # ── Gate controls — always visible ───────
    with col_gates:
        st.subheader("Gates")
        fw_enabled = st.checkbox("FW gate", value=False)
        if fw_enabled:
            fw_start = st.number_input("FW start (µs)", value=0.0, step=0.1, format="%.2f", key="fw_start")
            fw_end   = st.number_input("FW end (µs)",   value=2.0, step=0.1, format="%.2f", key="fw_end")
        st.divider()
        bw_enabled = st.checkbox("BW gate", value=False)
        if bw_enabled:
            bw_start = st.number_input("BW start (µs)", value=0.0, step=0.1, format="%.2f", key="bw_start")
            bw_end   = st.number_input("BW end (µs)",   value=2.0, step=0.1, format="%.2f", key="bw_end")
        st.divider()
        data_gate_enabled = st.checkbox("Data gate", value=False)
        if data_gate_enabled:
            dg_start = st.number_input("Data gate start (µs)", value=0.0,  step=0.1, format="%.2f", key="dg_start")
            dg_end   = st.number_input("Data gate end (µs)",   value=10.0, step=0.1, format="%.2f", key="dg_end")

    # ── Center waveform — loads when data ready ──
    with col_wave:
        st.subheader("Center waveform")
        if parsed_ok and dat_file:
            with st.spinner("Loading scan data..."):
                try:
                    waveform_data = load_cscan_data(dat_file, metadata)
                    data_loaded   = True
                except Exception as e:
                    st.error(f"Failed to load .dat file: {e}")
                    data_loaded = False
        else:
            data_loaded = False

        if data_loaded:
            time_axis      = metadata['time_axis']
            x_axis, y_axis = build_spatial_axes(metadata)
            center_wf      = get_center_waveform(waveform_data)

            # Gate defaults anchored to real time axis once data is loaded
            if fw_enabled:
                fw_start = st.session_state.get("fw_start", time_axis[0])
                fw_end   = st.session_state.get("fw_end",   time_axis[0] + 2.0)
            if bw_enabled:
                bw_start = st.session_state.get("bw_start", time_axis[-1] - 4.0)
                bw_end   = st.session_state.get("bw_end",   time_axis[-1] - 1.0)
            if data_gate_enabled:
                dg_start = st.session_state.get("dg_start", time_axis[0])
                dg_end   = st.session_state.get("dg_end",   time_axis[-1])

            fig_wf = go.Figure()
            fig_wf.add_trace(go.Scatter(
                x=time_axis, y=center_wf, mode='lines',
                line=dict(color='steelblue', width=1), name='Waveform'
            ))
            if fw_enabled:
                fig_wf.add_vrect(x0=fw_start, x1=fw_end,
                                 fillcolor="rgba(255,160,122,0.25)", line_width=1,
                                 line_color="tomato", annotation_text="FW",
                                 annotation_position="top left")
            if bw_enabled:
                fig_wf.add_vrect(x0=bw_start, x1=bw_end,
                                 fillcolor="rgba(144,238,144,0.25)", line_width=1,
                                 line_color="seagreen", annotation_text="BW",
                                 annotation_position="top left")
            if data_gate_enabled:
                fig_wf.add_vrect(x0=dg_start, x1=dg_end,
                                 fillcolor="rgba(180,180,180,0.2)", line_width=1,
                                 line_color="gray", annotation_text="Data",
                                 annotation_position="top left")
            fig_wf.update_layout(
                xaxis_title="Time (µs)", yaxis_title="Amplitude (normalized)",
                height=350, margin=dict(l=20, r=20, t=30, b=40), hovermode="x unified"
            )
            fig_wf.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                                 spikedash='dot', spikecolor='gray', spikethickness=1)
            fig_wf.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                                 spikedash='dot', spikecolor='gray', spikethickness=1)
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("Upload .txt and .dat files to view waveform.")

    st.divider()

    # ══════════════════════════════════════════
    # COLORMAP SECTION — always visible
    # ══════════════════════════════════════════
    st.subheader("C-scan")

    ctrl_col, thick_col, _ = st.columns([2, 2, 4])
    with ctrl_col:
        quantity_options = ["Peak amplitude (full window)"]
        if data_gate_enabled:
            quantity_options += ["Peak amplitude (gate)", "ToF (gate)", "Wave speed (gate)"]
        selected_quantity = st.selectbox("Plot quantity", quantity_options)
    with thick_col:
        if data_loaded and "Wave speed" in selected_quantity:
            thickness_mm = st.number_input("Sample thickness (mm)", value=10.0, step=0.1, format="%.3f")
            thickness_m  = thickness_mm * 1e-3
        else:
            st.markdown(" ")  # keep layout stable

    col_cmap, col_clicked_wf = st.columns([2, 1])

    with col_cmap:
        colormap_choice = st.selectbox(
            "Colormap", ["viridis", "plasma", "inferno", "magma", "gray", "RdBu", "jet"],
            key="cmap1"
        )
        if data_loaded:
            # Compute selected map
            if selected_quantity == "Peak amplitude (full window)":
                cmap_data  = compute_amplitude_map(waveform_data)
                cbar_label = "Amplitude"
            elif selected_quantity == "Peak amplitude (gate)":
                idx0       = time_to_index(dg_start, time_axis)
                idx1       = time_to_index(dg_end,   time_axis)
                cmap_data  = compute_peak_amplitude_map(waveform_data, idx0, idx1)
                cbar_label = "Amplitude"
            elif selected_quantity == "ToF (gate)":
                idx0       = time_to_index(dg_start, time_axis)
                idx1       = time_to_index(dg_end,   time_axis)
                cmap_data  = compute_tof_map(waveform_data, idx0, idx1, time_axis)
                cbar_label = "ToF (µs)"
            elif selected_quantity == "Wave speed (gate)":
                idx0       = time_to_index(dg_start, time_axis)
                idx1       = time_to_index(dg_end,   time_axis)
                tof_map    = compute_tof_map(waveform_data, idx0, idx1, time_axis)
                cmap_data  = compute_wave_speed_map(tof_map, thickness_m)
                cbar_label = "Wave speed (m/s)"

            fig_cmap = go.Figure(data=go.Heatmap(
                z=cmap_data, x=x_axis, y=y_axis,
                colorscale=colormap_choice,
                colorbar=dict(title=cbar_label),
                hovertemplate="x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>value: %{z:.4f}<extra></extra>"
            ))
            fig_cmap.update_layout(
                xaxis_title="Scan axis (mm)", yaxis_title="Index axis (mm)",
                height=420, margin=dict(l=20, r=20, t=30, b=40),
            )
            fig_cmap.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                                   spikedash='dot', spikecolor='white', spikethickness=1)
            fig_cmap.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                                   spikedash='dot', spikecolor='white', spikethickness=1)
            clicked = plotly_events(fig_cmap, click_event=True, key="cmap_click")
        else:
            st.info("Upload data to view C-scan.")
            clicked = []

    with col_clicked_wf:
        st.markdown("**Waveform at selected pixel**")
        if data_loaded and clicked:
            click_x_mm = clicked[0]['x']
            click_y_mm = clicked[0]['y']
            j = int(np.argmin(np.abs(x_axis - click_x_mm)))
            i = int(np.argmin(np.abs(y_axis - click_y_mm)))
            i = int(np.clip(i, 0, waveform_data.shape[0] - 1))
            j = int(np.clip(j, 0, waveform_data.shape[1] - 1))

            selected_wf  = waveform_data[i, j, :]
            selected_env = compute_envelope(selected_wf)

            fig_sel = go.Figure()
            fig_sel.add_trace(go.Scatter(x=time_axis, y=selected_wf, mode='lines',
                                         line=dict(color='steelblue', width=1), name='Waveform'))
            fig_sel.add_trace(go.Scatter(x=time_axis, y=selected_env, mode='lines',
                                         line=dict(color='tomato', width=1, dash='dash'), name='Envelope'))
            if fw_enabled:
                fig_sel.add_vrect(x0=fw_start, x1=fw_end, fillcolor="rgba(255,160,122,0.2)",
                                  line_width=1, line_color="tomato")
            if bw_enabled:
                fig_sel.add_vrect(x0=bw_start, x1=bw_end, fillcolor="rgba(144,238,144,0.2)",
                                  line_width=1, line_color="seagreen")
            if data_gate_enabled:
                fig_sel.add_vrect(x0=dg_start, x1=dg_end, fillcolor="rgba(180,180,180,0.15)",
                                  line_width=1, line_color="gray")
            fig_sel.update_layout(
                xaxis_title="Time (µs)", yaxis_title="Amplitude",
                height=420, margin=dict(l=10, r=10, t=30, b=40),
                legend=dict(orientation="h", y=1.1),
                title=f"Pixel ({j}, {i}) — {click_x_mm:.2f} mm, {click_y_mm:.2f} mm"
            )
            st.plotly_chart(fig_sel, use_container_width=True)
        elif data_loaded:
            st.info("Click a pixel on the C-scan to view its waveform here.")
        else:
            st.info("Upload data first.")

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
            fig_csv.add_trace(go.Scatter(x=sample_index, y=waveform, mode='lines',
                                          line=dict(color='steelblue', width=1), name='Waveform'))
            fig_csv.update_layout(xaxis_title="Sample index", yaxis_title="Amplitude",
                                   height=350, margin=dict(l=20, r=20, t=30, b=40),
                                   hovermode="x unified")
            st.plotly_chart(fig_csv, use_container_width=True)
        with col_result:
            st.subheader("ToF measurement")
            fs_input = st.number_input("Sampling frequency (MHz)", value=125.0, step=1.0)
            st.info("Gate-based ToF measurement coming next.")
    else:
        st.info("Upload a single waveform CSV file to begin.")