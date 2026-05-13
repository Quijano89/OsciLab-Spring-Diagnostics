def run_analysis(csv_path, m, px_per_inch):

    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter, find_peaks

    data = pd.read_csv(csv_path)

    t = data["t"].values.astype(float)
    y = data["y"].values.astype(float)

    y = -y

    dt = np.mean(np.diff(t))

    window = min(11, len(y)//2*2 - 1)
    window = max(window, 5)

    y_smooth = savgol_filter(y, window, 3)

    # -------------------------
    # INCH CALIBRATION (PIXELS → METERS)
    # -------------------------
    m_per_px = 0.0254 / px_per_inch
    y0 = y_smooth * m_per_px
    y0 = y0 - np.mean(y0)

    trend = np.polyfit(t, y0, 1)
    y0 = y0 - (trend[0] * t + trend[1])

    v = np.gradient(y0, dt)

    # -------------------------
    # FFT
    # -------------------------
    fft_vals = np.fft.fft(y0)
    freqs = np.fft.fftfreq(len(y0), dt)

    mask = freqs > 0

    f = freqs[mask][np.argmax(np.abs(fft_vals[mask]))]
    omega = 2 * np.pi * f

    # -------------------------
    # PEAKS
    # -------------------------
    peaks, _ = find_peaks(
        y0,
        distance=max(2, int(0.3 / dt))
    )

    peaks_t = t[peaks]
    peaks_y = np.abs(y0[peaks])

    # -------------------------
    # DAMPING
    # -------------------------
    gamma = 0

    if len(peaks_t) >= 2:
        gamma = -np.polyfit(
            peaks_t,
            np.log(peaks_y + 1e-12),
            1
        )[0]

    zeta = gamma / np.sqrt(omega**2 + gamma**2)

    # -------------------------
    # SPRING CONSTANT
    # -------------------------
    k = m * omega**2

    # -------------------------
    # MODEL
    # -------------------------
    y_model = np.exp(-gamma * t) * np.cos(omega * t) * np.max(np.abs(y0))

    # -------------------------
    # ENERGY
    # -------------------------
    E = 0.5 * v**2 + 0.5 * omega**2 * y0**2

    # -------------------------
    # FIT QUALITY
    # -------------------------
    rmse = np.sqrt(np.mean((y0 - y_model)**2))
    signal_range = np.max(y0) - np.min(y0)
    nrmse = rmse / signal_range if signal_range != 0 else 0

    # -------------------------
    # RESULTS
    # -------------------------
    results = {
        "f": f,
        "omega": omega,
        "gamma": gamma,
        "zeta": zeta,
        "k": k,
        "nrmse": nrmse
    }

    # -------------------------
    # PLOTS DATA
    # -------------------------
    plots = {
        "t": t,
        "y0": y0,
        "y_model": y_model,
        "E": E,
        "v": v,
        "peaks_t": peaks_t,
        "peaks_y": peaks_y
    }

    # -------------------------
    # CSV EXPORT (NEW ADDITION ONLY)
    # -------------------------
    export_df = pd.DataFrame({
        "t": t,
        "y_m": y0,
        "v_m": v,
        "E": E,
        "y_model": y_model
    })

    results_df = pd.DataFrame([results])

    return results, plots, export_df, results_df