import sys
sys.modules['tensorboard'] = None

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import os

st.set_page_config(
    page_title='IEIS — Autonomous Industrial Energy Intelligence System',
    page_icon='⚙️',
    layout='wide'
)

# ── Constants ─────────────────────────────────────────────────────────────────

MNAMES = ['CNC_Turning', 'CNC_Milling', 'Induction_Hardener',
          'Hydraulic_Press', 'Cylindrical_Grinder']

MACHINE_DISPLAY = {
    'CNC_Turning':         {'label': 'CNC Turning Centre',     'power': 15.0, 'sensor': 'IFM VVB001',       'cost': '₹20,000'},
    'CNC_Milling':         {'label': 'CNC Milling Centre',     'power': 11.0, 'sensor': 'IFM VVB001',       'cost': '₹20,000'},
    'Induction_Hardener':  {'label': 'Induction Hardener',     'power': 18.0, 'sensor': 'Schneider PM5000', 'cost': '₹15,000'},
    'Hydraulic_Press':     {'label': 'Hydraulic Spline Press', 'power': 22.0, 'sensor': 'Schneider PM5000', 'cost': '₹15,000'},
    'Cylindrical_Grinder': {'label': 'Cylindrical Grinder',    'power': 7.5,  'sensor': 'IFM VVB001',       'cost': '₹20,000'},
}

MACHINE_POWER_KW = {1: 15.0, 2: 11.0, 3: 18.0, 4: 22.0, 5: 7.5}
TARIFF_TO_MULT   = {6.0: 0.75, 8.0: 1.00, 11.0: 1.375}
SLOT_HOURS       = 0.25


def build_tariff_cycle():
    cycle = []
    for h in range(24):
        t = 11.0 if 18 <= h < 22 else (6.0 if h >= 22 or h < 6 else 8.0)
        cycle.extend([t] * 4)
    return cycle


DAILY_TARIFF = build_tariff_cycle()

# ── Gymnasium Environment ─────────────────────────────────────────────────────

class DriveShaftProductionEnv(gym.Env):
    """
    Five-machine drive-shaft production cell.
    Sequential constraint: jobs must pass through machines 1→2→3→4→5 in order.
    Observation: [tariff_mult, health×5, jobs_remaining, time_pressure]
    Action: 0=WAIT, 1-5=route to machine
    """
    metadata = {'render_modes': []}

    def __init__(self, n_jobs=10):
        super().__init__()
        self.n_jobs    = n_jobs
        self.max_steps = 200
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.3, 1.3, 1.3, 1.3, 1.3, float(n_jobs), 1.0], dtype=np.float32)
        )
        self.health = {m: 1.0 for m in range(1, 6)}
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.job_stages   = [0] * self.n_jobs
        self.completed    = 0

    def _get_obs(self):
        t  = DAILY_TARIFF[self.current_step % 96]
        tm = TARIFF_TO_MULT[t]
        hs = [self.health[m] for m in range(1, 6)]
        return np.array(
            [tm] + hs + [float(self.n_jobs - self.completed),
                         self.current_step / self.max_steps],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self.health = {m: 1.0 for m in range(1, 6)}
        return self._get_obs(), {}

    def step(self, action):
        t      = DAILY_TARIFF[self.current_step % 96]
        tm     = TARIFF_TO_MULT[t]
        reward = -0.10

        if action == 0:
            reward -= 0.05
        elif 1 <= action <= 5:
            mid   = action
            h     = self.health[mid]
            kw    = MACHINE_POWER_KW[mid]
            stage = mid - 1
            elig  = [j for j in range(self.n_jobs) if self.job_stages[j] == stage]

            if elig:
                reward -= kw * tm * h * SLOT_HOURS * 0.5
                if h > 1.08:
                    reward -= 30.0 * (h - 1.0)
                self.job_stages[elig[0]] += 1
                if self.job_stages[elig[0]] >= 5:
                    self.completed += 1
                    reward += 120.0
            else:
                reward -= 10.0

        self.current_step += 1
        done = False

        if self.completed >= self.n_jobs:
            reward += 600.0
            done    = True
        elif self.current_step >= self.max_steps:
            reward -= 80.0 * (self.n_jobs - self.completed)
            done    = True

        return self._get_obs(), reward, done, False, {
            'completed':      self.completed,
            'jobs_remaining': self.n_jobs - self.completed,
        }

# ── Data and Model Loading ────────────────────────────────────────────────────

@st.cache_data
def load_dataset():
    path = 'IEIS_Master_Dataset_Final_v2.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_resource
def load_model():
    # Try both possible filenames
    for fname in ['IEIS_PPO_Agent.zip', 'IEIS_v3_PPO_Agent.zip',
                  'IEIS_PPO_Agent', 'IEIS_v3_PPO_Agent']:
        if os.path.exists(fname):
            try:
                return PPO.load(fname)
            except Exception:
                continue
    return None

# ── Simulation Helpers ────────────────────────────────────────────────────────

def run_sim(model, health_scenario, use_fifo=False):
    env = DriveShaftProductionEnv(n_jobs=10)
    obs, _ = env.reset()
    env.health = health_scenario.copy()
    obs = env._get_obs()

    log, done, step = [], False, 0
    while not done:
        if use_fifo:
            action = (step % 5) + 1
        else:
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

        obs, _, done, _, info = env.step(action)
        log.append({
            'action':    action,
            'tariff':    DAILY_TARIFF[step % 96],
            'health':    env.health.get(action, 1.0) if action >= 1 else 1.0,
            'completed': info['completed'],
        })
        step += 1

    return log, info


def cost_inr(log):
    return round(
        sum(MACHINE_POWER_KW[e['action']] * e['tariff'] * SLOT_HOURS * e['health']
            for e in log if 1 <= e['action'] <= 5),
        2
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image('https://img.icons8.com/fluency/96/manufacturing.png', width=70)
st.sidebar.title('IEIS')
st.sidebar.markdown('**Autonomous Industrial Energy Intelligence System**')
st.sidebar.markdown('---')

press_health = st.sidebar.select_slider(
    'Set Press Health State for Demo',
    options=[1.00, 1.05, 1.10, 1.15, 1.20],
    value=1.20,
    format_func=lambda v: (
        f'{v:.2f}x — Nominal'          if v <= 1.00 else
        f'{v:.2f}x — B_Acceptable'     if v <= 1.05 else
        f'{v:.2f}x — Approaching Alarm' if v <= 1.10 else
        f'{v:.2f}x — C_Alarm'          if v <= 1.15 else
        f'{v:.2f}x — D_Danger'
    )
)

run_btn = st.sidebar.button('▶  Run Simulation', type='primary')
st.sidebar.markdown('---')
st.sidebar.markdown('**Component:** AISI 4140 Steel Drive-Shaft')
st.sidebar.markdown('**Facility:** Chakan Industrial Belt, Pune')
st.sidebar.markdown('**Sensor investment:** ₹90,000')

# ── Load resources ────────────────────────────────────────────────────────────

df    = load_dataset()
model = load_model()

HEALTH_SCENARIO = {1: 1.00, 2: 1.00, 3: 1.00, 4: press_health, 5: 1.00}

# Run simulation once and cache in session state so all tabs share results
if run_btn:
    if model is None:
        st.session_state['sim_error'] = True
        st.session_state['sim_done']  = False
    else:
        with st.spinner('Running simulation...'):
            ai_log, ai_info   = run_sim(model, HEALTH_SCENARIO, use_fifo=False)
            fi_log, fi_info   = run_sim(model, HEALTH_SCENARIO, use_fifo=True)

        ai_c  = cost_inr(ai_log)
        fi_c  = cost_inr(fi_log)
        sav   = max(0, fi_c - ai_c)
        pct   = round(sav / fi_c * 100, 1) if fi_c > 0 else 0
        co2   = round((sav / 8.0) * 0.82, 1)

        st.session_state.update({
            'sim_done':    True,
            'sim_error':   False,
            'ai_log':      ai_log,
            'fi_log':      fi_log,
            'ai_info':     ai_info,
            'fi_info':     fi_info,
            'ai_c':        ai_c,
            'fi_c':        fi_c,
            'sav':         sav,
            'pct':         pct,
            'co2':         co2,
            'press_health': press_health,
        })

sim_done  = st.session_state.get('sim_done', False)
sim_error = st.session_state.get('sim_error', False)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    '🏭 System Configuration',
    '📊 Executive Dashboard',
    '🤖 AI Decision Centre',
    '💰 Financial Audit',
])

# ── TAB 1: System Configuration ───────────────────────────────────────────────

with tab1:
    st.header('Workstation Configuration — AISI 4140 Drive-Shaft Production Cell')
    st.markdown(
        '**Component:** Rear axle half-shaft for B-segment passenger vehicles '
        '(Maruti Swift, Honda Amaze). 40 units per 8-hour shift.'
    )

    st.subheader('Machine & Sensor Specifications')
    spec_rows = []
    for mn, md in MACHINE_DISPLAY.items():
        iso = 'A_Nominal'
        if df is not None:
            pr = df[df[f'{mn}_Producing'] == True]
            if len(pr) > 0:
                iso = pr[f'{mn}_ISO_Status'].mode()[0]
        icon = '🟢' if iso == 'A_Nominal' else ('🟡' if iso == 'B_Acceptable' else '🔴')
        spec_rows.append({
            'Machine':       md['label'],
            'Rated Power':   f"{md['power']}kW",
            'Sensor':        md['sensor'],
            'Sensor Cost':   md['cost'],
            'Annual Status': f"{icon} {iso}",
        })
    st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)

    st.subheader('Sensor ROI — 5-Year Cost of Inaction vs IEIS-Monitored')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Run-to-Failure (No Sensors)**')
        st.markdown('''| Year | Event | Cost |
| --- | --- | --- |
| Y1-2 | Friction waste — 22kW × 1.15 × 2 shifts/day | ₹1,04,000 |
| Y3 | Unplanned bearing failure | ₹1,40,000 |
| Y4 | Second failure | ₹1,40,000 |
| Y5 | End-of-life replacement | ₹3,50,000 |
| **Total** | | **₹7,34,000** |''')
    with c2:
        st.markdown('**IEIS-Monitored**')
        st.markdown('''| Year | Event | Cost |
| --- | --- | --- |
| Y1 | Sensor investment | ₹90,000 |
| Y1-5 | AI scheduling savings | −₹2,62,500 |
| Y1-5 | Planned maintenance (no emergency) | ₹2,10,000 |
| Y5 | End-of-life replacement | ₹3,50,000 |
| **Total** | | **₹3,87,500** |''')
    st.success('**IEIS saves ₹3,46,500 over 5 years on the Hydraulic Press alone.**')

# ── TAB 2: Executive Dashboard ────────────────────────────────────────────────

with tab2:
    st.header('Executive Dashboard — Factory Health & Shift Status')

    if df is not None:
        st.subheader('Machine Health — Full Year Summary')
        cols = st.columns(5)
        for idx, (mn, md) in enumerate(MACHINE_DISPLAY.items()):
            pr   = df[df[f'{mn}_Producing'] == True]
            iso  = pr[f'{mn}_ISO_Status'].mode()[0] if len(pr) > 0 else 'A_Nominal'
            vib  = pr[f'{mn}_Vib_RMS'].dropna().mean() if len(pr) > 0 else 0.0
            kurt = pr[f'{mn}_Vib_Kurtosis'].dropna().mean() if len(pr) > 0 else 0.0
            icon = '🟢' if iso == 'A_Nominal' else ('🟡' if iso == 'B_Acceptable' else '🔴')
            cols[idx].metric(
                md['label'][:13],
                f'{icon} {iso}',
                f'RMS:{vib:.3f} K:{kurt:.1f}'
            )

    if sim_error:
        st.error(
            'Model file not found on the server. '
            'Make sure **IEIS_PPO_Agent.zip** is in the GitHub repository root.'
        )
    elif sim_done:
        ai_c  = st.session_state['ai_c']
        fi_c  = st.session_state['fi_c']
        sav   = st.session_state['sav']
        pct   = st.session_state['pct']
        co2   = st.session_state['co2']
        ai_info = st.session_state['ai_info']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('FIFO Cost',     f'₹{fi_c:,.0f}')
        c2.metric('AI Cost',       f'₹{ai_c:,.0f}', delta=f'-₹{sav:,.0f}')
        c3.metric('Shift Saving',  f'{pct}%')
        c4.metric('Carbon Offset', f'{co2} kg CO₂')

        on_time = '✅ ON-TIME' if ai_info['completed'] == 10 else '❌ BREACH'
        st.info(f'**Delivery:** {on_time}  |  Jobs completed: {ai_info["completed"]}/10')
    else:
        st.info('Set the Press health state in the sidebar and click **▶ Run Simulation**.')

# ── TAB 3: AI Decision Centre ─────────────────────────────────────────────────

with tab3:
    st.header('AI Decision Centre — Scheduling Logic')

    if sim_done:
        ai_log = st.session_state['ai_log']
        actions = [e['action'] for e in ai_log]
        tariffs = [e['tariff'] for e in ai_log]

        CMAP    = {0: '#9E9E9E', 1: '#1565C0', 2: '#0288D1',
                   3: '#2E7D32', 4: '#B71C1C', 5: '#6A1B9A'}
        MLABELS = {0: 'WAIT', 1: 'Turning', 2: 'Milling',
                   3: 'Hardener', 4: 'Press', 5: 'Grinder'}

        fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        axes[0].plot(tariffs, color='crimson', linewidth=1.8)
        axes[0].set_ylabel('Tariff (₹/kWh)')
        axes[0].set_title('Real-Time Electricity Tariff vs AI Scheduling Decisions')
        axes[0].grid(alpha=0.25)

        for t, a in enumerate(actions):
            axes[1].bar(t, 1, color=CMAP.get(a, 'grey'), width=1, alpha=0.85)
        patches = [mpatches.Patch(color=CMAP[k], label=MLABELS[k]) for k in CMAP]
        axes[1].legend(handles=patches, loc='upper right', fontsize=8)
        axes[1].set_yticks([])
        axes[1].set_xlabel('Timestep (15-min slots)')
        st.pyplot(fig)
        plt.close(fig)

        # Summary stats
        wait_count  = sum(1 for a in actions if a == 0)
        press_count = sum(1 for a in actions if a == 4)
        col1, col2, col3 = st.columns(3)
        col1.metric('Total Steps',            len(actions))
        col2.metric('Strategic WAITs',        wait_count,
                    help='Agent deferred jobs to avoid peak tariff')
        col3.metric('Press Routings',         press_count,
                    help='Lower = better when press is degraded')

        st.subheader('Decision Audit Trail — Last 15 Actions')
        MNAME_FULL = {
            0: 'WAIT', 1: 'CNC Turning', 2: 'CNC Milling',
            3: 'Induction Hardener', 4: 'Hydraulic Press', 5: 'Cylindrical Grinder'
        }
        for i, e in enumerate(ai_log[-15:]):
            a = e['action']
            t = e['tariff']
            h = e.get('health', 1.0)
            if a == 0:
                reason = f'Tariff at ₹{t:.0f}/kWh — deferring to cheaper window'
            elif a == 4 and h > 1.08:
                reason = f'Press routed despite health penalty {h:.2f}x — no alternative available'
            else:
                reason = f'Optimal routing | Tariff ₹{t:.0f}/kWh | Jobs done: {e["completed"]}'
            st.markdown(f'**Step {len(ai_log)-15+i+1}** `{MNAME_FULL[a]}` — {reason}')
    else:
        st.info('Run the simulation from the Executive Dashboard tab.')

# ── TAB 4: Financial Audit ────────────────────────────────────────────────────

with tab4:
    st.header('Financial Audit — Business Impact')

    if sim_done:
        ai_c    = st.session_state['ai_c']
        fi_c    = st.session_state['fi_c']
        sav     = st.session_state['sav']
        pct     = st.session_state['pct']
        ph      = st.session_state['press_health']
        annual  = round(sav * 250, 0)
        payback = round(90000 / (annual / 250)) if annual > 0 else 9999

        c1, c2, c3 = st.columns(3)
        c1.metric('Annual Saving',  f'₹{annual:,.0f}')
        c2.metric('Sensor Payback', f'{payback} shifts')
        c3.metric('5-Year Net',     f'₹{annual * 5 - 90000:,.0f}')

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            ['FIFO\n(Manual Scheduling)', 'PPO Agent\n(IEIS)'],
            [fi_c, ai_c],
            color=['#EF5350', '#42A5F5'],
            edgecolor='black',
            width=0.45
        )
        ax.set_ylabel('Shift Energy Cost (₹)')
        ax.set_title(f'Cost Comparison — Press Health {ph:.2f}x | {pct}% Saving Achieved')
        for bar, val in zip(bars, [fi_c, ai_c]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'₹{val:.0f}',
                ha='center', fontweight='bold', fontsize=11
            )
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('---')
        st.markdown(f'''
**Summary**
- Sensor total investment: **₹90,000**
- Saving this shift: **₹{sav:.0f}** ({pct}%)
- Projected annual saving: **₹{annual:,.0f}** (250 working days)
- Sensor payback period: **{payback} production shifts**
- Carbon offset this shift: **{round((sav/8)*0.82, 1)} kg CO₂**
- 5-year net benefit: **₹{annual*5-90000:,.0f}**
        ''')

        # 5-year ROI bar chart
        years      = list(range(1, 6))
        cumulative = [annual * y - 90000 for y in years]
        colours    = ['#EF5350' if v < 0 else '#66BB6A' for v in cumulative]

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.bar(years, cumulative, color=colours, edgecolor='black', width=0.5)
        ax2.axhline(0, color='black', linewidth=1.2)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Net Benefit (₹)')
        ax2.set_title('5-Year ROI — Sensor Investment vs Cumulative Savings')
        ax2.grid(alpha=0.3, axis='y')
        for i, v in enumerate(cumulative):
            ax2.text(i + 1, v + (2000 if v >= 0 else -6000),
                     f'₹{v:,.0f}', ha='center', fontsize=8, fontweight='bold')
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info('Run the simulation from the Executive Dashboard tab.')