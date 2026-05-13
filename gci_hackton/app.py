import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import pandas as pd   # <-- ADDED

from tracking import run_tracking
from analysis import run_analysis

# -------------------------
# STYLE (UNCHANGED)
# -------------------------
DARK_BG   = "#0d0d0f"
PANEL_BG  = "#13131a"
GRID_CLR  = "#1e1e2e"
TICK_CLR  = "#4a4a6a"
TEXT_CLR  = "#c8c8e0"
ACCENT1   = "#7ef0c8"
ACCENT2   = "#f07e9a"
ACCENT3   = "#7eb8f0"
ACCENT4   = "#f0c87e"

def apply_style():
    rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_CLR,
        "axes.labelcolor": TEXT_CLR,
        "xtick.color": TICK_CLR,
        "ytick.color": TICK_CLR,
        "font.family": "monospace",
    })

def make_fig():
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax

# -------------------------
# APP
# -------------------------
st.title("OsciLab: Spring Diagnostics")

video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

m = st.number_input("Mass (kg)", min_value=0.0, value=1.0, step=1.0)

if video_file:

    video_path = "temp.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success("Video uploaded")

    if st.button("Run Tracking"):
        df, fps, px_per_inch = run_tracking(video_path)

        st.session_state["px_per_inch"] = px_per_inch
        st.session_state["tracking_done"] = True

        st.success(f"Tracking done: {len(df)} points")

    if st.button("Run Analysis"):

        if "px_per_inch" not in st.session_state:
            st.error("Run tracking first.")
        else:

            results, plots, export_df, results_df = run_analysis(
                "centers.csv",
                m,
                st.session_state["px_per_inch"]
            )

            st.subheader("Results")

            st.metric("f (Hz)", results["f"])
            st.metric("ω (rad/s)", results["omega"])
            st.metric("γ (1/s)", results["gamma"])
            st.metric("ζ", results["zeta"])
            st.metric("k", results["k"])
            st.metric("NRMSE", results["nrmse"])

            # ---------------- DOWNLOAD CSV (ADDED)
            st.download_button(
                "Download Data CSV",
                export_df.to_csv(index=False),
                file_name="oscillation_data.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download Summary CSV",
                results_df.to_csv(index=False),
                file_name="oscillation_summary.csv",
                mime="text/csv"
            )

            # ---------------- Plot 1
            fig, ax = make_fig()
            ax.set_title("Displacement vs Time", color="white")
            ax.plot(plots["t"], plots["y0"], color=ACCENT1)
            ax.plot(plots["t"], plots["y_model"], color=ACCENT2, linestyle="--")
            st.pyplot(fig)

            # ---------------- Plot 2
            fig, ax = make_fig()
            ax.set_title("Energy vs Time", color="white")
            ax.plot(plots["t"], plots["E"], color=ACCENT3)
            st.pyplot(fig)

            # ---------------- Plot 3
            fig, ax = make_fig()
            ax.set_title("Phase Space", color="white")
            ax.plot(plots["y0"], plots["v"], color=ACCENT4)
            st.pyplot(fig)

            # ---------------- Plot 4
            fig, ax = make_fig()
            ax.set_title("Amplitude Envelope", color="white")
            ax.scatter(plots["peaks_t"], plots["peaks_y"], color=ACCENT4)

            if len(plots["peaks_t"]) >= 2:
                env = np.exp(
                    np.polyval(
                        np.polyfit(
                            plots["peaks_t"],
                            np.log(plots["peaks_y"] + 1e-12),
                            1
                        ),
                        plots["peaks_t"]
                    )
                )
                ax.plot(plots["peaks_t"], env, color=ACCENT2, linestyle="--")

            st.pyplot(fig)