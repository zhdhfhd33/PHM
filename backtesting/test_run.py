"""
Phase 3 backtesting script - PHM system vs baseline comparison (Test Run)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = Path(__file__).parent
_PROJ_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.simulation.engine import SimulationEngine
from src.simulation.comparator import BaselineSimulator
from src.optimization.scheduler import solve_maintenance_schedule, threshold_schedule

# ── simulation parameters ────────────────────────────
N_AGV = 10
K_SLOTS = 2
N_STEPS = 500
MILP_INTERVAL = 24
SEED = 42
# ──────────────────────────────────────────────────────

def run_backtesting():
    np.random.seed(SEED)

    print("=" * 70)
    print("  PHM System vs Multiple Baselines Backtesting (TEST RUN)")
    print(f"  AGVs: {N_AGV}, Maintenance bays (K): {K_SLOTS}")
    print(f"  Simulation period: {N_STEPS}h ({N_STEPS/24:.1f} days)")
    print("=" * 70)

    # 1. Initialize PHM and B1
    phm_engine = SimulationEngine(num_agvs=N_AGV, K=K_SLOTS)
    initial_ruls = [agv.rul_mean for agv in phm_engine.agvs]

    b1_engine = BaselineSimulator(num_agvs=N_AGV, mode='NO_MAINT', K=K_SLOTS, initial_ruls=initial_ruls)
    # 2. Prepare B2 candidates
    tbm_candidates = [200, 500]
    b2_engines = {f"B2 TBM ({interval}h)": BaselineSimulator(num_agvs=N_AGV, mode='TBM', tbm_interval=interval, K=K_SLOTS, initial_ruls=initial_ruls)
                  for interval in tbm_candidates}

    # 3. Prepare B3 Pred-CBM candidates
    b3_thresholds = [48, 192]
    b3_engines = {
        f"B3 Pred-CBM ({thr}h)": SimulationEngine(num_agvs=N_AGV, K=K_SLOTS, initial_ruls=initial_ruls)
        for thr in b3_thresholds
    }
    b3_threshold_map = {f"B3 Pred-CBM ({thr}h)": thr for thr in b3_thresholds}

    all_engines = {
        'PHM': phm_engine,
        'B1': b1_engine,
        **b2_engines,
        **b3_engines,
    }

    cost_histories = {name: [] for name in all_engines}

    # 4. Run simulation loop
    print(f"  Running simulation for {N_STEPS} steps...")
    for t in range(N_STEPS):
        for name, eng in all_engines.items():
            eng.step()

        if phm_engine.current_time % MILP_INTERVAL == 0:
            rul_samples = {agv.id: agv.rul_samples for agv in phm_engine.agvs}
            in_progress = phm_engine.get_in_progress_dict()
            schedule, _ = solve_maintenance_schedule(
                rul_samples, in_progress=in_progress, N_AGV=N_AGV, K=K_SLOTS,
                T=168
            )
            schedule_24h = {k: v for k, v in schedule.items()
                            if v is not None and v < MILP_INTERVAL}
            phm_engine.set_schedule(schedule_24h)

            for name, b3_eng in b3_engines.items():
                thr = b3_threshold_map[name]
                b3_in_progress = b3_eng.get_in_progress_ids()
                b3_schedule = threshold_schedule(b3_eng.agvs, thr, K_SLOTS, b3_in_progress)
                b3_eng.set_schedule(b3_schedule)

        for name, eng in all_engines.items():
            cost_histories[name].append(eng.cost_accumulated)

    # 5. Final metrics calculation
    print("  Simulation complete. Calculating metrics...")
    phm_metrics = phm_engine.get_full_metrics()
    print(f"  PHM Cost: {phm_metrics['cost']}")

    return phm_metrics

if __name__ == "__main__":
    run_backtesting()
