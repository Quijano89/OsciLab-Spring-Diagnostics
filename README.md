# OsciLab: Spring Diagnostics

A physics-based computer vision system that extracts and analyzes damped oscillatory motion from video data. The project combines **computer vision, signal processing, and physics modeling** to estimate key parameters of a spring-mass system.

---

## 🧠 What it does

Given a video of a swinging or oscillating object, OsciLab:

- Tracks the object using OpenCV-based tracking
- Extracts position vs time data
- Converts pixel motion into physical motion (calibrated using mass and scaling)
- Estimates system parameters:
  - Frequency (f)
  - Angular frequency (ω)
  - Damping coefficient (γ)
  - Damping ratio (ζ)
  - Spring constant (k)
- Evaluates model accuracy using NRMSE
- Visualizes motion, energy, phase space, and damping envelope

---

## 📊 Outputs

The system generates:

1. **Displacement vs Time**
   - Raw motion vs fitted damped oscillator model

2. **Energy vs Time**
   - Total mechanical energy evolution

3. **Phase Space Plot**
   - Velocity vs displacement dynamics

4. **Amplitude Envelope**
   - Exponential decay of oscillation peaks

---

## ⚙️ How it works

1. **Tracking**
   - Object tracked frame-by-frame using OpenCV tracker
   - Center position extracted per frame

2. **Signal Processing**
   - Noise smoothing (Savitzky–Golay filter)
   - Drift removal
   - Velocity estimation via numerical differentiation

3. **Frequency Extraction**
   - FFT used to estimate dominant oscillation frequency

4. **Damping Estimation**
   - Logarithmic decay of peak amplitudes

5. **Model Fitting**
   - Damped harmonic oscillator model:
     \[
     x(t) = A e^{-\gamma t} \cos(\omega t)
     \]

---

## 📐 Physics Model

- Angular frequency:
  \[
  \omega = 2\pi f
  \]

- Spring constant:
  \[
  k = m\omega^2
  \]

- Damping ratio:
  \[
  \zeta = \frac{\gamma}{\sqrt{\omega^2 + \gamma^2}}
  \]

---

## 🖥️ Run Locally

```bash
git clone https://github.com/Quijano89/OsciLab-Spring-Diagnostics.git
cd OsciLab-Spring-Diagnostics
pip install -r requirements.txt
streamlit run app.py
