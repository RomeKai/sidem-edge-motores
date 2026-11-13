"""Monte Carlo comparison of maintenance strategies on a fleet of motors.

Every number here is an explicit, documented ASSUMPTION (see CostModel and FleetAssumptions)
so the report can discuss sensitivity instead of presenting them as facts.

Failure modes per motor:
    - wear-out: Weibull(shape, scale) life, detectable in advance (P-F interval).
    - abrupt: Poisson events, NOT predictable; only the hard real-time guard can limit damage.

Strategies:
    corrective   - run to failure. No edge node.
    preventive   - fixed-interval replacement. No edge node.
    predictive   - soft alerts trigger an immediate planned intervention; hard guard active.
    prescriptive - predictive + recommended action: derate load to extend RUL, schedule the
                   intervention in the next planned production stop, targeted repair from
                   the diagnosis, and fewer false-alarm interventions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRATEGIES = ["corrective", "preventive", "predictive", "prescriptive"]


@dataclass(frozen=True)
class CostModel:
    downtime_cost_per_h: float = 200.0  # lost production, USD/h
    planned_repair: float = 400.0
    planned_downtime_h: float = 4.0
    targeted_repair: float = 250.0  # prescriptive: diagnosis points to the failing part
    failure_repair: float = 1500.0  # unplanned: rewinding/replacement + collateral damage
    failure_downtime_h: float = 24.0
    guarded_repair: float = 600.0  # abrupt event stopped by hard guard before major damage
    guarded_downtime_h: float = 8.0
    inspection_cost: float = 100.0  # false alarm: technician checks, finds nothing
    inspection_downtime_h: float = 1.0

    def planned_cost(self) -> float:
        return self.planned_repair + self.planned_downtime_h * self.downtime_cost_per_h

    def failure_cost(self) -> float:
        return self.failure_repair + self.failure_downtime_h * self.downtime_cost_per_h


@dataclass(frozen=True)
class FleetAssumptions:
    weibull_shape: float = 2.5  # >1: wear-out regime
    weibull_scale_days: float = 900.0  # ~2.5 years characteristic life
    abrupt_rate_per_year: float = 0.5
    preventive_interval_days: float = 540.0
    detection_prob: float = 0.9  # soft detector recall for wear-out failures
    lead_days: tuple[float, float] = (7.0, 20.0)  # P-F interval (alert -> failure)
    false_alarms_per_year: float = 1.0
    prescriptive_false_alarm_filter: float = 0.5  # fraction of false alarms dismissed by diagnosis
    derate_rul_gain: float = 0.3  # derating extends remaining life by 30%
    planned_stop_every_days: float = 7.0
    planned_stop_downtime_factor: float = 0.25  # production loss when done in a planned stop


def _simulate_motor(strategy: str, horizon: float, rng: np.random.Generator, c: CostModel, a: FleetAssumptions) -> dict:
    m = {"failures": 0, "planned_interventions": 0, "downtime_h": 0.0, "total_cost": 0.0}
    edge = strategy in ("predictive", "prescriptive")

    def add(cost: float, downtime: float, planned: bool) -> None:
        m["total_cost"] += cost + downtime * c.downtime_cost_per_h
        m["downtime_h"] += downtime
        m["planned_interventions" if planned else "failures"] += 1

    # Abrupt events are independent of renewals (external causes) -> pre-sample them.
    n_abrupt = rng.poisson(a.abrupt_rate_per_year * horizon / 365.0)
    for _ in range(n_abrupt):
        if edge:
            add(c.guarded_repair, c.guarded_downtime_h, planned=False)
        else:
            add(c.failure_repair, c.failure_downtime_h, planned=False)

    if edge:
        fa_rate = a.false_alarms_per_year
        if strategy == "prescriptive":
            fa_rate *= 1.0 - a.prescriptive_false_alarm_filter
        for _ in range(rng.poisson(fa_rate * horizon / 365.0)):
            add(c.inspection_cost, c.inspection_downtime_h, planned=True)

    t = 0.0
    while t < horizon:
        life = a.weibull_scale_days * rng.weibull(a.weibull_shape)
        fail_at = t + life
        detected = rng.random() < a.detection_prob
        lead = rng.uniform(*a.lead_days)

        if strategy == "corrective":
            if fail_at >= horizon:
                break
            add(c.failure_repair, c.failure_downtime_h, planned=False)
            t = fail_at
        elif strategy == "preventive":
            replace_at = t + a.preventive_interval_days
            if fail_at < replace_at:
                if fail_at >= horizon:
                    break
                add(c.failure_repair, c.failure_downtime_h, planned=False)
                t = fail_at
            else:
                if replace_at >= horizon:
                    break
                add(c.planned_repair, c.planned_downtime_h, planned=True)
                t = replace_at
        else:
            alert_at = fail_at - lead
            if not detected or alert_at < t:
                if fail_at >= horizon:
                    break
                add(c.failure_repair, c.failure_downtime_h, planned=False)
                t = fail_at
                continue
            if alert_at >= horizon:
                break
            if strategy == "predictive":
                add(c.planned_repair, c.planned_downtime_h, planned=True)
                t = alert_at
            else:
                rul = lead * (1.0 + a.derate_rul_gain)
                wait = a.planned_stop_every_days - (alert_at % a.planned_stop_every_days)
                if wait < rul - 1.0:
                    downtime = c.planned_downtime_h * a.planned_stop_downtime_factor
                    t = alert_at + wait
                else:
                    downtime = c.planned_downtime_h
                    t = alert_at
                add(c.targeted_repair, downtime, planned=True)
    return m


def simulate_fleet(
    n_motors: int = 200,
    horizon_days: float = 3 * 365,
    seed: int = 0,
    costs: CostModel = CostModel(),
    assumptions: FleetAssumptions = FleetAssumptions(),
) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        agg = {"failures": 0, "planned_interventions": 0, "downtime_h": 0.0, "total_cost": 0.0}
        for motor in range(n_motors):
            # common random numbers: each motor sees the same random stream under every strategy
            rng = np.random.default_rng([seed, motor])
            for k, v in _simulate_motor(strategy, horizon_days, rng, costs, assumptions).items():
                agg[k] += v
        rows.append({"strategy": strategy, **agg})
    return pd.DataFrame(rows)
