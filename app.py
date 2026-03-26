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
    compute_peak_amplitude_map,
    compute_tof_map,
    compute_envelope,
    time_to_index,
    build_spatial_axes,
    compute_wave_speed_xcorr,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Ultrasound NDE", layout="wide")
st.markdown("<h1 style='text-align: center;'>Ultrasound NDE Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
mode = st.radio("", ["C-scan (2D raster)", "Single waveform (CSV)"], horizontal=True, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)
# st.divider()

# ═════════════════════════════════════════════
# MODE 1 — C-SCAN
# ═════════════════════════════════════════════
if mode == "C-scan (2D raster)":

    # ── File upload ───────────────────────────
    col_dat, col_txt = st.columns(2)
    with col_dat:
        dat_file = st.file_uploader("Data file (.dat)", type=["dat"])
    with col_txt:
        txt_file = st.file_uploader("Metadata file (.txt)", type=["txt"])

    st.divider()

    # ── Parse metadata (top level) ────────────
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

    # ── Load data (top level) ─────────────────
    if parsed_ok and dat_file:
        with st.spinner("Loading scan data..."):
            try:
                waveform_data  = load_cscan_data(dat_file, metadata)
                data_loaded    = True
                time_axis      = metadata['time_axis']
                x_axis, y_axis = build_spatial_axes(metadata)
                # Center spatial axes at zero
                x_axis = x_axis - x_axis[-1] / 2.0
                y_axis = y_axis - y_axis[-1] / 2.0
            except Exception as e:
                st.error(f"Failed to load .dat file: {e}")
                data_loaded = False
    else:
        data_loaded = False

    # ── Clear state when no data loaded ──────
    if not data_loaded:
        for key in ['sel_i', 'sel_j', 'cmap1_data', 'cmap1_label',
                    'cmap2_data', 'cmap2_label',
                    'fw_start', 'fw_end', 'bw_start', 'bw_end',
                    'dg_start', 'dg_end']:
            st.session_state.pop(key, None)

    # ══════════════════════════════════════════
    # ROW 1 — Info | Waveform | Gates
    # ══════════════════════════════════════════
    col_info, col_main = st.columns([1, 4])

    # ── LEFT SIDEBAR: Acquisition + Gates ────
    with col_info:
        st.subheader("Gates")

        t0 = float(time_axis[0])  if data_loaded else 0.0
        t1 = float(time_axis[-1]) if data_loaded else 10.0

        gcol_cb, gcol_s, gcol_e = st.columns([1, 1, 1])
        with gcol_cb:
            fw_enabled = st.checkbox("FW gate", value=False)
        with gcol_s:
            fw_s_def = st.session_state.get('fw_start', t0)
            fw_start = st.number_input("Start (µs)", value=fw_s_def, step=0.1, format="%.2f",
                                       key="fw_start_input", disabled=not fw_enabled,
                                       label_visibility="collapsed" if not fw_enabled else "visible")
        with gcol_e:
            fw_e_def = st.session_state.get('fw_end', t0 + 2.0)
            fw_end   = st.number_input("End (µs)", value=fw_e_def, step=0.1, format="%.2f",
                                       key="fw_end_input", disabled=not fw_enabled,
                                       label_visibility="collapsed" if not fw_enabled else "visible")
        if fw_enabled:
            st.session_state['fw_start'] = fw_start
            st.session_state['fw_end']   = fw_end
        else:
            fw_start = st.session_state.get('fw_start', None)
            fw_end   = st.session_state.get('fw_end',   None)

        gcol_cb, gcol_s, gcol_e = st.columns([1, 1, 1])
        with gcol_cb:
            bw_enabled = st.checkbox("BW gate", value=False)
        with gcol_s:
            bw_s_def = st.session_state.get('bw_start', t1 - 4.0)
            bw_start = st.number_input("Start (µs)", value=bw_s_def, step=0.1, format="%.2f",
                                       key="bw_start_input", disabled=not bw_enabled,
                                       label_visibility="collapsed" if not bw_enabled else "visible")
        with gcol_e:
            bw_e_def = st.session_state.get('bw_end', t1 - 1.0)
            bw_end   = st.number_input("End (µs)", value=bw_e_def, step=0.1, format="%.2f",
                                       key="bw_end_input", disabled=not bw_enabled,
                                       label_visibility="collapsed" if not bw_enabled else "visible")
        if bw_enabled:
            st.session_state['bw_start'] = bw_start
            st.session_state['bw_end']   = bw_end
        else:
            bw_start = st.session_state.get('bw_start', None)
            bw_end   = st.session_state.get('bw_end',   None)

        gcol_cb, gcol_s, gcol_e = st.columns([1, 1, 1])
        with gcol_cb:
            dg_enabled = st.checkbox("Data gate", value=False)
        with gcol_s:
            dg_s_def = st.session_state.get('dg_start', t0)
            dg_start = st.number_input("Start (µs)", value=dg_s_def, step=0.1, format="%.2f",
                                       key="dg_start_input", disabled=not dg_enabled,
                                       label_visibility="collapsed" if not dg_enabled else "visible")
        with gcol_e:
            dg_e_def = st.session_state.get('dg_end', t1)
            dg_end   = st.number_input("End (µs)", value=dg_e_def, step=0.1, format="%.2f",
                                       key="dg_end_input", disabled=not dg_enabled,
                                       label_visibility="collapsed" if not dg_enabled else "visible")
        if dg_enabled:
            st.session_state['dg_start'] = dg_start
            st.session_state['dg_end']   = dg_end
        else:
            dg_start = st.session_state.get('dg_start', None)
            dg_end   = st.session_state.get('dg_end',   None)

        st.subheader("Acquisition")

        def show_val(label, key, unit=""):
            val = metadata.get(key, "—")
            display = f"{val} {unit}" if (val != "—" and unit) else str(val)
            st.markdown(f"**{label}:** {display}")

        show_val("Transducer", "transducer", "MHz")
        show_val("Voltage",    "voltage",    "V")
        show_val("Damping",    "damping",    "Ω")
        show_val("Gain",       "gain",       "dB")
        st.subheader("Scan geometry")
        show_val("Scan points",     "scan_points")
        show_val("Index points",    "index_points")
        show_val("Scan increment",  "scan_increment",  "mm")
        show_val("Index increment", "index_increment", "mm")
        tcol_label, tcol_input = st.columns([1, 2])
        with tcol_label:
            st.markdown("**Thickness (mm)**")
        with tcol_input:
            thickness_raw = st.text_input(
                "Thickness", value="10.000",
                key="thickness_mm", label_visibility="collapsed"
            )
        try:
            thickness_m = float(thickness_raw) * 1e-3
        except ValueError:
            st.error("Enter a valid number for thickness.")
            st.stop()
        st.subheader("Sampling")
        show_val("Fs",          "f_sampling",       "MHz")
        show_val("Samples/wfm", "number_of_samples")
        show_val("Start time",  "start_time",       "µs")
        show_val("Window",      "length_of_signal", "µs")

        

    # ── RIGHT MAIN AREA ───────────────────────
    with col_main:

        # ── TOP: Waveform ─────────────────────
        if data_loaded:
            sel_i = st.session_state.get('sel_i', waveform_data.shape[0] // 2)
            sel_j = st.session_state.get('sel_j', waveform_data.shape[1] // 2)

            st.markdown("<h3 style='text-align: center;'>Waveform</h3>", unsafe_allow_html=True)

            display_wf = waveform_data[sel_i, sel_j, :]
            envelope   = compute_envelope(display_wf)

            fig_wf = go.Figure()
            fig_wf.add_trace(go.Scatter(
                x=time_axis, y=display_wf, mode='lines',
                line=dict(color='steelblue', width=2), name='RF'
            ))
            # fig_wf.add_trace(go.Scatter(
            #     x=time_axis, y=envelope, mode='lines',
            #     line=dict(color='tomato', width=1, dash='dash'), name='Envelope'
            # ))
            if fw_enabled and fw_start is not None:
                fig_wf.add_vrect(x0=fw_start, x1=fw_end,
                                 fillcolor="rgba(255,160,122,0.25)", line_width=1,
                                 line_color="tomato", annotation_text="FW",
                                 annotation_position="top left")
            if bw_enabled and bw_start is not None:
                fig_wf.add_vrect(x0=bw_start, x1=bw_end,
                                 fillcolor="rgba(144,238,144,0.25)", line_width=1,
                                 line_color="seagreen", annotation_text="BW",
                                 annotation_position="top left")
            if dg_enabled and dg_start is not None:
                fig_wf.add_vrect(x0=dg_start, x1=dg_end,
                                 fillcolor="rgba(180,180,180,0.2)", line_width=1,
                                 line_color="gray", annotation_text="Data",
                                 annotation_position="top left")
            fig_wf.update_layout(
                xaxis_title="Time (µs)",
                yaxis_title="Amplitude (normalized)",
                height=400,
                margin=dict(l=60, r=30, t=30, b=60),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(
                    title_font=dict(size=16, color="black"),
                    tickfont=dict(size=14, color="black"),
                    linecolor="black", linewidth=1.5,
                    mirror=True,
                    showgrid=True, gridcolor="rgba(200,200,200,0.4)",
                    zeroline=False,
                    showspikes=True, spikemode='across', spikesnap='cursor',
                    spikedash='dot', spikecolor='gray', spikethickness=1,
                ),
                yaxis=dict(
                    title_font=dict(size=16, color="black"),
                    tickfont=dict(size=14, color="black"),
                    linecolor="black", linewidth=1.5,
                    mirror=True,
                    showgrid=True, gridcolor="rgba(200,200,200,0.4)",
                    zeroline=True, zerolinecolor="rgba(150,150,150,0.5)", zerolinewidth=1,
                    showspikes=True, spikemode='across', spikesnap='cursor',
                    spikedash='dot', spikecolor='gray', spikethickness=1,
                ),
            )
        else:
            st.subheader("Center waveform")
            st.info("Upload .txt and .dat files to view waveform.")

        st.divider()

        # ── BOTTOM: Two colormaps ─────────────
        st.subheader("C-scan")

        CMAP_RANGE     = 15.0
        CMAP_OPTIONS   = ["viridis", "plasma", "inferno", "magma", "gray", "RdBu", "jet"]
        QTY_OPTIONS    = ["Peak amplitude", "ToF", "Wave speed (xcorr)"]

        WIN_OPTIONS_BASE = ["Full window"]
        if fw_enabled and fw_start is not None: WIN_OPTIONS_BASE.append("FW gate")
        if bw_enabled and bw_start is not None: WIN_OPTIONS_BASE.append("BW gate")
        if dg_enabled and dg_start is not None: WIN_OPTIONS_BASE.append("Data gate")


        def gate_indices(window):
            if not data_loaded:
                return 0, 1
            if window == "Full window":
                return 0, len(time_axis)
            elif window == "FW gate":
                return time_to_index(fw_start, time_axis), time_to_index(fw_end, time_axis)
            elif window == "BW gate":
                return time_to_index(bw_start, time_axis), time_to_index(bw_end, time_axis)
            elif window == "Data gate":
                return time_to_index(dg_start, time_axis), time_to_index(dg_end, time_axis)
            return 0, len(time_axis)

        def compute_map(quantity, window):
            idx0, idx1 = gate_indices(window)
            if quantity == "Peak amplitude":
                return compute_peak_amplitude_map(waveform_data, idx0, idx1), "Amplitude"
            elif quantity == "ToF":
                return compute_tof_map(waveform_data, idx0, idx1, time_axis), "ToF (µs)"
            elif quantity == "Wave speed (xcorr)":
                if not (fw_enabled and bw_enabled and
                        fw_start is not None and bw_start is not None):
                    st.error("Wave speed (xcorr) requires both FW and BW gates to be set.")
                    st.stop()
                ws, _ = compute_wave_speed_xcorr(
                    waveform_data,
                    fw_gate_us         = (fw_start, fw_end),
                    bw_gate_us         = (bw_start, bw_end),
                    time_axis          = time_axis,
                    f_sampling_hz      = metadata['f_sampling_hz'],
                    sample_interval_us = metadata['sample_interval'],
                    thickness_m        = thickness_m,
                )
                return ws, "Wave speed (m/s)"

        def make_cmap_fig(cmap_data, cbar_label, colorscale, marker_i, marker_j):
            fig = go.Figure(data=go.Heatmap(
                z=cmap_data,
                x=np.round(x_axis, 4).tolist(),
                y=np.round(y_axis, 4).tolist(),
                colorscale=colorscale,
                colorbar=dict(title=cbar_label, thickness=14),
                hovertemplate="x: %{x:.2f} mm<br>y: %{y:.2f} mm<br>value: %{z:.4f}<extra></extra>"
            ))
            if marker_i is not None and marker_j is not None:
                fig.add_trace(go.Scatter(
                    x=[x_axis[marker_j]], y=[y_axis[marker_i]],
                    mode='markers',
                    marker=dict(symbol='cross', size=12, color='red',
                                line=dict(width=2, color='white')),
                    showlegend=False
                ))
            fig.update_layout(
                xaxis_title="Scan axis (mm)",
                yaxis_title="Index axis (mm)",
                height=500,
                margin=dict(l=20, r=20, t=30, b=40),
                xaxis=dict(range=[-CMAP_RANGE, CMAP_RANGE], constrain='domain'),
                yaxis=dict(range=[-CMAP_RANGE, CMAP_RANGE],
                           scaleanchor='x', scaleratio=1),
            )
            fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                             spikedash='dot', spikecolor='white', spikethickness=1)
            fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                             spikedash='dot', spikecolor='white', spikethickness=1)
            return fig

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            # st.markdown("**C-scan 1**")
            c1a, c1b, c1c, c1d = st.columns([2, 2, 2, 1])
            with c1a:
                cc1_qty  = st.selectbox("Quantity", QTY_OPTIONS,      key="cc1_qty")
            with c1b:
                cc1_win = st.selectbox("Window", WIN_OPTIONS_BASE, key="cc1_win",
                                       disabled=(cc1_qty == "Wave speed (xcorr)"))
            with c1c:
                cc1_cmap = st.selectbox("Colormap", CMAP_OPTIONS,     key="cc1_cmap")
            with c1d:
                st.markdown("<br>", unsafe_allow_html=True)
                upd1 = st.button("▶", key="upd1", use_container_width=True)

            if data_loaded and (upd1 or 'cmap1_data' not in st.session_state):
                with st.spinner("Computing C-scan 1..."):
                    try:
                        d, lbl = compute_map(cc1_qty, cc1_win)
                        st.session_state['cmap1_data']  = d
                        st.session_state['cmap1_label'] = lbl
                    except Exception as e:
                        st.error(f"C-scan 1 error: {e}")

            if data_loaded and 'cmap1_data' in st.session_state:
                fig1 = make_cmap_fig(
                    st.session_state['cmap1_data'],
                    st.session_state['cmap1_label'],
                    cc1_cmap,
                    st.session_state.get('sel_i'),
                    st.session_state.get('sel_j'),
                )
                ev1 = st.plotly_chart(fig1, use_container_width=True,
                                      on_select="rerun", key="cmap1_plot")
                if ev1 and ev1.selection and ev1.selection.points:
                    pt    = ev1.selection.points[0]
                    new_j = int(np.clip(np.argmin(np.abs(x_axis - pt['x'])), 0, waveform_data.shape[1]-1))
                    new_i = int(np.clip(np.argmin(np.abs(y_axis - pt['y'])), 0, waveform_data.shape[0]-1))
                    if new_i != st.session_state.get('sel_i') or new_j != st.session_state.get('sel_j'):
                        st.session_state['sel_i'] = new_i
                        st.session_state['sel_j'] = new_j
                        st.rerun()
                # st.caption("Click a pixel to update the waveform above.")
            else:
                st.info("Upload data and press **▶ Update** to plot.")

        with col_c2:
            # st.markdown("**C-scan 2**")
            c2a, c2b, c2c, c2d = st.columns([2, 2, 2, 1])
            with c2a:
                cc2_qty  = st.selectbox("Quantity", QTY_OPTIONS,      key="cc2_qty",  index=1)
            with c2b:
                cc2_win = st.selectbox("Window", WIN_OPTIONS_BASE, key="cc2_win",
                                       disabled=(cc2_qty == "Wave speed (xcorr)"))
            with c2c:
                cc2_cmap = st.selectbox("Colormap", CMAP_OPTIONS,     key="cc2_cmap", index=1)
            with c2d:
                st.markdown("<br>", unsafe_allow_html=True)
                upd2 = st.button("▶", key="upd2", use_container_width=True)

            if data_loaded and (upd2 or 'cmap2_data' not in st.session_state):
                with st.spinner("Computing C-scan 2..."):
                    try:
                        d, lbl = compute_map(cc2_qty, cc2_win)
                        st.session_state['cmap2_data']  = d
                        st.session_state['cmap2_label'] = lbl
                    except Exception as e:
                        st.error(f"C-scan 2 error: {e}")

            if data_loaded and 'cmap2_data' in st.session_state:
                fig2 = make_cmap_fig(
                    st.session_state['cmap2_data'],
                    st.session_state['cmap2_label'],
                    cc2_cmap,
                    st.session_state.get('sel_i'),
                    st.session_state.get('sel_j'),
                )
                ev2 = st.plotly_chart(fig2, use_container_width=True,
                                      on_select="rerun", key="cmap2_plot")
                if ev2 and ev2.selection and ev2.selection.points:
                    pt    = ev2.selection.points[0]
                    new_j = int(np.clip(np.argmin(np.abs(x_axis - pt['x'])), 0, waveform_data.shape[1]-1))
                    new_i = int(np.clip(np.argmin(np.abs(y_axis - pt['y'])), 0, waveform_data.shape[0]-1))
                    if new_i != st.session_state.get('sel_i') or new_j != st.session_state.get('sel_j'):
                        st.session_state['sel_i'] = new_i
                        st.session_state['sel_j'] = new_j
                        st.rerun()
                # st.caption("Click a pixel to update the waveform above.")
            else:
                st.info("Upload data and press **▶ Update** to plot.")

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
            st.number_input("Sampling frequency (MHz)", value=125.0, step=1.0)
            st.info("Gate-based ToF measurement coming next.")
    else:
        st.info("Upload a single waveform CSV file to begin.")