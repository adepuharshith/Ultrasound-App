import streamlit as st
import numpy as np
import plotly.graph_objects as go
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

    # ── File upload ───────────────────────────
    col_upload_left, col_upload_right = st.columns([2, 2])
    with col_upload_left:
        txt_file = st.file_uploader("Upload metadata file (.txt)", type=["txt"])
    with col_upload_right:
        dat_file = st.file_uploader("Upload data file (.dat)", type=["dat"])

    st.divider()

    # ── Parse metadata ────────────────────────
    # Done at top level so metadata is available everywhere below
    if txt_file:
        try:
            metadata  = parse_metadata_from_txt(txt_file)
            parsed_ok = True
        except Exception as e:
            st.error(f"Metadata error: {e}")
            parsed_ok = False
            metadata  = {}
    else:
        parsed_ok = False
        metadata  = {}

    # ── Load waveform data ────────────────────
    # Also at top level so waveform_data is available to both waveform and colormap sections
    if parsed_ok and dat_file:
        with st.spinner("Loading scan data..."):
            try:
                waveform_data = load_cscan_data(dat_file, metadata)
                data_loaded   = True
                time_axis     = metadata['time_axis']
                x_axis, y_axis = build_spatial_axes(metadata)
            except Exception as e:
                st.error(f"Failed to load .dat file: {e}")
                data_loaded = False
    else:
        data_loaded = False

    # ══════════════════════════════════════════
    # TOP SECTION: Info | Waveform | Gates
    # ══════════════════════════════════════════
    col_info, col_wave, col_gates = st.columns([1, 3, 1])

    # ── Acquisition parameters ────────────────
    with col_info:
        st.subheader("Acquisition")

        def show_val(label, key, unit=""):
            val = metadata.get(key, "—")
            display = f"{val} {unit}" if val != "—" and unit else str(val)
            st.markdown(f"**{label}:** {display}")

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
        show_val("Fs",          "f_sampling",       "MHz")
        show_val("Samples/wfm", "number_of_samples")
        show_val("Start time",  "start_time",       "µs")
        show_val("Window",      "length_of_signal", "µs")

    # ── Gate controls ─────────────────────────
    with col_gates:
        st.subheader("Gates")

        fw_enabled = st.checkbox("FW gate", value=False)
        if fw_enabled:
            t0_fw    = float(time_axis[0])   if data_loaded else 0.0
            fw_start = st.number_input("FW start (µs)", value=t0_fw,       step=0.1, format="%.2f", key="fw_start")
            fw_end   = st.number_input("FW end (µs)",   value=t0_fw + 2.0, step=0.1, format="%.2f", key="fw_end")
        else:
            fw_start = fw_end = None

        st.divider()
        bw_enabled = st.checkbox("BW gate", value=False)
        if bw_enabled:
            t1_bw    = float(time_axis[-1]) if data_loaded else 10.0
            bw_start = st.number_input("BW start (µs)", value=t1_bw - 4.0, step=0.1, format="%.2f", key="bw_start")
            bw_end   = st.number_input("BW end (µs)",   value=t1_bw - 1.0, step=0.1, format="%.2f", key="bw_end")
        else:
            bw_start = bw_end = None

        st.divider()
        dg_enabled = st.checkbox("Data gate", value=False)
        if dg_enabled:
            t0_dg  = float(time_axis[0])  if data_loaded else 0.0
            t1_dg  = float(time_axis[-1]) if data_loaded else 10.0
            dg_start = st.number_input("Data gate start (µs)", value=t0_dg, step=0.1, format="%.2f", key="dg_start")
            dg_end   = st.number_input("Data gate end (µs)",   value=t1_dg, step=0.1, format="%.2f", key="dg_end")
        else:
            dg_start = dg_end = None

    # ── Center waveform ───────────────────────
    with col_wave:
        st.subheader("Center waveform")
        if data_loaded:
            center_wf = get_center_waveform(waveform_data)
            fig_wf    = go.Figure()
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
            if dg_enabled:
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
    # BOTTOM SECTION: C-scan colormap
    # ══════════════════════════════════════════
    st.subheader("C-scan")

    # ── Controls row ─────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 1])

    with ctrl1:
        selected_quantity = st.selectbox("Plot quantity", ["Peak amplitude", "ToF", "Wave speed"])

    with ctrl2:
        window_options = ["Full window"]
        if fw_enabled:  window_options.append("FW gate")
        if bw_enabled:  window_options.append("BW gate")
        if dg_enabled:  window_options.append("Data gate")
        selected_window = st.selectbox("Plot window", window_options)

    with ctrl3:
        if selected_quantity == "Wave speed":
            thickness_mm = st.number_input("Sample thickness (mm)", value=10.0, step=0.01, format="%.3f")
            thickness_m  = thickness_mm * 1e-3
        else:
            thickness_m  = None
            st.markdown(" ")

    with ctrl4:
        st.markdown("<br>", unsafe_allow_html=True)
        update_btn = st.button("▶  Update C-scan", use_container_width=True)

    # ── Compute map on button press or first load ──
    if data_loaded:
        # Determine gate indices
        def get_gate_indices(window):
            if window == "Full window":
                return 0, len(time_axis)
            elif window == "FW gate" and fw_enabled:
                return time_to_index(fw_start, time_axis), time_to_index(fw_end, time_axis)
            elif window == "BW gate" and bw_enabled:
                return time_to_index(bw_start, time_axis), time_to_index(bw_end, time_axis)
            elif window == "Data gate" and dg_enabled:
                return time_to_index(dg_start, time_axis), time_to_index(dg_end, time_axis)
            return 0, len(time_axis)

        should_compute = update_btn or ('cmap_data' not in st.session_state)

        if should_compute:
            with st.spinner("Computing C-scan..."):
                try:
                    idx0, idx1 = get_gate_indices(selected_window)

                    if selected_quantity == "Peak amplitude":
                        cmap_data  = compute_peak_amplitude_map(waveform_data, idx0, idx1)
                        cbar_label = "Amplitude"
                    elif selected_quantity == "ToF":
                        cmap_data  = compute_tof_map(waveform_data, idx0, idx1, time_axis)
                        cbar_label = "ToF (µs)"
                    elif selected_quantity == "Wave speed":
                        tof_map    = compute_tof_map(waveform_data, idx0, idx1, time_axis)
                        cmap_data  = compute_wave_speed_map(tof_map, thickness_m)
                        cbar_label = "Wave speed (m/s)"

                    st.session_state['cmap_data']  = cmap_data
                    st.session_state['cbar_label'] = cbar_label

                except Exception as e:
                    st.error(f"Error computing map: {e}")

    else:
        # Clear cache if no data loaded
        if 'cmap_data' in st.session_state:
            del st.session_state['cmap_data']
            del st.session_state['cbar_label']

    # ── Colormap plot + pixel waveform ─────────
    col_cmap, col_pixel_wf = st.columns([2, 1])

    with col_cmap:
        colormap_choice = st.selectbox(
            "Colormap",
            ["viridis", "plasma", "inferno", "magma", "gray", "RdBu", "jet"],
            key="cmap1"
        )

        if data_loaded and 'cmap_data' in st.session_state:
            cmap_data  = st.session_state['cmap_data']
            cbar_label = st.session_state['cbar_label']

            fig_cmap = go.Figure(data=go.Heatmap(
                z=cmap_data,
                x=np.round(x_axis, 4).tolist(),
                y=np.round(y_axis, 4).tolist(),
                colorscale=colormap_choice,
                colorbar=dict(title=cbar_label, thickness=15),
                hovertemplate="x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>value: %{z:.4f}<extra></extra>"
            ))
            fig_cmap.update_layout(
                xaxis_title="Scan axis (mm)",
                yaxis_title="Index axis (mm)",
                height=450,
                margin=dict(l=20, r=20, t=30, b=40),
            )
            fig_cmap.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                                   spikedash='dot', spikecolor='white', spikethickness=1)
            fig_cmap.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                                   spikedash='dot', spikecolor='white', spikethickness=1)
            st.plotly_chart(fig_cmap, use_container_width=True)
        else:
            st.info("Upload data and press **▶ Update C-scan** to plot." if data_loaded else "Upload data to view C-scan.")

    # ── Pixel waveform viewer ──────────────────
    with col_pixel_wf:
        st.markdown("**Waveform at selected pixel**")
        if data_loaded:
            max_i = waveform_data.shape[0] - 1
            max_j = waveform_data.shape[1] - 1
            sel_i = st.slider("Index axis (row)",  0, max_i, max_i // 2, key="sel_i")
            sel_j = st.slider("Scan axis (col)",   0, max_j, max_j // 2, key="sel_j")

            # Show physical coordinates
            x_mm = x_axis[sel_j]
            y_mm = y_axis[sel_i]
            st.caption(f"Position: x = {x_mm:.2f} mm, y = {y_mm:.2f} mm")

            selected_wf  = waveform_data[sel_i, sel_j, :]
            selected_env = compute_envelope(selected_wf)

            fig_sel = go.Figure()
            fig_sel.add_trace(go.Scatter(
                x=time_axis, y=selected_wf, mode='lines',
                line=dict(color='steelblue', width=1), name='RF'
            ))
            fig_sel.add_trace(go.Scatter(
                x=time_axis, y=selected_env, mode='lines',
                line=dict(color='tomato', width=1, dash='dash'), name='Envelope'
            ))
            if fw_enabled:
                fig_sel.add_vrect(x0=fw_start, x1=fw_end,
                                  fillcolor="rgba(255,160,122,0.2)",
                                  line_width=1, line_color="tomato")
            if bw_enabled:
                fig_sel.add_vrect(x0=bw_start, x1=bw_end,
                                  fillcolor="rgba(144,238,144,0.2)",
                                  line_width=1, line_color="seagreen")
            if dg_enabled:
                fig_sel.add_vrect(x0=dg_start, x1=dg_end,
                                  fillcolor="rgba(180,180,180,0.15)",
                                  line_width=1, line_color="gray")
            fig_sel.update_layout(
                xaxis_title="Time (µs)", yaxis_title="Amplitude",
                height=420,
                margin=dict(l=10, r=10, t=40, b=40),
                legend=dict(orientation="h", y=1.1),
                title=f"Pixel ({sel_j}, {sel_i})"
            )
            st.plotly_chart(fig_sel, use_container_width=True)
        else:
            st.info("Upload data to inspect pixel waveforms.")

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
                x=sample_index, y=waveform, mode='lines',
                line=dict(color='steelblue', width=1), name='Waveform'
            ))
            fig_csv.update_layout(
                xaxis_title="Sample index", yaxis_title="Amplitude",
                height=350, margin=dict(l=20, r=20, t=30, b=40),
                hovermode="x unified"
            )
            st.plotly_chart(fig_csv, use_container_width=True)

        with col_result:
            st.subheader("ToF measurement")
            fs_input = st.number_input("Sampling frequency (MHz)", value=125.0, step=1.0)
            st.info("Gate-based ToF measurement coming next.")
    else:
        st.info("Upload a single waveform CSV file to begin.")