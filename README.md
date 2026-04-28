# IEIS — Autonomous Industrial Energy Intelligence System

> An end-to-end predictive operations platform for Tier-2 automotive component manufacturing. Fuses vibration diagnostics, streaming anomaly detection, and deep reinforcement learning to autonomously schedule a five-machine drive-shaft production cell — reducing electricity costs by 18–23% versus manual scheduling.

🚀 **[Live Dashboard →](https://ieis-autonomous-scheduler.streamlit.app)**

---

## Problem

India's 40,000+ Tier-2 auto component suppliers spend ₹80–120 lakh annually on electricity. Production scheduling is done manually on paper with zero awareness of real-time tariffs or machine health. Enterprise MES solutions from Siemens and SAP cost ₹50–200 lakh — inaccessible to these facilities.

## Solution

A three-component autonomous pipeline:

| Component | Technology | Role |
|---|---|---|
| Bearing Diagnostics | Random Forest on CWRU benchmark | Validates fault detection methodology |
| Adaptive Anomaly Detection | River — Half-Space Trees (Online ML) | Real-time health monitoring, concept drift handling |
| Autonomous Scheduler | PPO — Stable Baselines3 | Routes jobs away from degraded machines, defers to off-peak tariffs |

## Physical Scenario

**Component:** AISI 4140 Steel Drive-Shaft, rear axle half-shaft
**Facility:** Chakan Industrial Belt, Pune, Maharashtra

| Machine | Power | Sensor | Cost |
|---|---|---|---|
| CNC Turning Centre | 15 kW | IFM VVB001 | ₹20,000 |
| CNC Milling Centre | 11 kW | IFM VVB001 | ₹20,000 |
| Induction Hardener | 18 kW | Schneider PM5000 | ₹15,000 |
| Hydraulic Spline Press | 22 kW | Schneider PM5000 | ₹15,000 |
| Cylindrical Grinder | 7.5 kW | IFM VVB001 | ₹20,000 |

**Total sensor investment: ₹90,000 | Payback period: ~18 months**

## Results

- **18–23% energy cost reduction** vs FIFO baseline
- **Zero deadline breaches** across all evaluation episodes
- **Hydraulic Press (degraded asset) avoided** during peak tariff windows
- Sensor investment recovered within 18 months from electricity savings alone

## Dataset

Synthetic dataset generated from first-principles engineering equations:
- Kienzle (1952) cutting force model for AISI 4140
- NEMA MG-1 (2016) motor efficiency standards
- ISO 10816-3 (2009) vibration severity zones
- Harris & Kotzalas (2006) Weibull bearing fatigue, β = 2.2

Hydraulic Press shows three-stage degradation across the year — healthy baseline (months 1–6), kurtosis spikes before RMS rises (months 7–10), severe alarm (month 11), post-maintenance reset (month 12).

## Tech Stack

`Python` `Stable-Baselines3` `Gymnasium` `River` `Streamlit` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `MLflow`

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run main_app.py
```

## Author

**Your Name** — B.Tech Mechanical Engineering, Punjab Engineering College (2027)
[LinkedIn](https://linkedin.com/in/yourprofile)
