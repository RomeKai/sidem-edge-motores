from si40sim.maintenance import CostModel, simulate_fleet

STRATEGIES = ["corrective", "preventive", "predictive", "prescriptive"]


def test_fleet_simulation_returns_all_strategies_with_metrics():
    res = simulate_fleet(n_motors=50, horizon_days=365, seed=1)
    assert list(res["strategy"]) == STRATEGIES
    for col in ["failures", "planned_interventions", "downtime_h", "total_cost"]:
        assert col in res.columns
        assert (res[col] >= 0).all()


def test_fleet_simulation_is_reproducible():
    a = simulate_fleet(n_motors=30, horizon_days=365, seed=5)
    b = simulate_fleet(n_motors=30, horizon_days=365, seed=5)
    assert a.equals(b)


def test_corrective_has_no_planned_interventions():
    res = simulate_fleet(n_motors=30, horizon_days=365, seed=2).set_index("strategy")
    assert res.loc["corrective", "planned_interventions"] == 0


def test_cost_model_failure_is_more_expensive_than_planned():
    c = CostModel()
    assert c.failure_cost() > c.planned_cost()
