import numpy as np
import gurobipy as gp
from gurobipy import GRB

gp.setParam("LogToConsole", 0)

# Auction function – same logic as your Model 3

def simulate_capex_auction(
    base_capex_per_mw,
    n_buyers,
    competition_intensity=0.10,
    saturation_scale=10.0,
    noise_level=0.05,
    seed=None,
):
    
    if seed is not None:
        np.random.seed(seed)

    base_capex_per_mw = np.asarray(base_capex_per_mw, dtype=float)

    if n_buyers <= 1:
        return base_capex_per_mw.copy()

    delta = n_buyers - 1
    factor = 1.0 + competition_intensity * (1.0 - np.exp(-delta / saturation_scale))
    noise = 1.0 + noise_level * np.random.uniform(-1.0, 1.0, size=len(base_capex_per_mw))

    return base_capex_per_mw * factor * noise



# Two-stage stochastic model with uncertainty via scenarios

def solve_model4(
    lambdas_scen,       # (S, T): λ_{s,t}
    c,                  # (K, T): c_{k,t}
    D_scen,             # (S, T): D_{s,t}
    CF_wind_scen,       # (S, T): CF_wind_{s,t}
    C_capex_per_MW,     # (K,): CAPEX per MW
    x_ub,               # (K,): tech-specific capacity upper bounds
    p_s,                # (S,): scenario probabilities
    alpha,              # (K,): availability factors
    X_max=None,         # scalar or None
    tech_names=None,    # list of strings, length K
):
    """
    Two-stage stochastic model:

    Stage 1 (here-and-now):
        - Choose capacities x[k] [MW] (same across scenarios).

    Stage 2 (recourse, per scenario and hour):
        - Dispatch y[k,t,s] [MWh] to meet demand in each scenario.

    Objective:
        Maximise expected profit:
            Σ_s p_s Σ_{t,k} (λ_{s,t} - c_{k,t}) y[k,t,s] - Σ_k C_k x_k.
    """

    lambdas_scen = np.asarray(lambdas_scen, dtype=float)
    D_scen       = np.asarray(D_scen, dtype=float)
    CF_wind_scen = np.asarray(CF_wind_scen, dtype=float)
    c            = np.asarray(c, dtype=float)
    C_capex_per_MW = np.asarray(C_capex_per_MW, dtype=float)
    x_ub         = np.asarray(x_ub, dtype=float)
    p_s          = np.asarray(p_s, dtype=float)
    alpha        = np.asarray(alpha, dtype=float)

    S, T = lambdas_scen.shape
    assert D_scen.shape == (S, T)
    assert CF_wind_scen.shape == (S, T)
    K, T_c = c.shape
    assert T_c == T
    assert C_capex_per_MW.shape == (K,)
    assert x_ub.shape == (K,)
    assert p_s.shape == (S,)
    assert alpha.shape == (K,)
    assert np.isclose(p_s.sum(), 1.0), "Scenario probabilities must sum to 1."

    if tech_names is None:
        tech_names = [f"tech_{k}" for k in range(K)]
    else:
        assert len(tech_names) == K

    hours = range(T)
    techs = range(K)
    scens = range(S)

    lower_names = [name.lower() for name in tech_names]
    idx_wind = lower_names.index("wind")

    m = gp.Model("Model4_uncertainty")

    # Stage 1: capacity decisions
    x = m.addVars(techs, lb=0.0, name="x")

    # Stage 2: dispatch decisions
    y = m.addVars(techs, hours, scens, lb=0.0, name="y")

    # Demand balance per scenario and hour
    for s in scens:
        for t in hours:
            m.addConstr(
                gp.quicksum(y[k, t, s] for k in techs) == D_scen[s, t],
                name=f"demand_balance[s={s},t={t}]",
            )

    # Capacity constraints
    for k in techs:
        for s in scens:
            for t in hours:
                if k == idx_wind:
                    m.addConstr(
                        y[k, t, s] <= x[k] * alpha[k] * CF_wind_scen[s, t],
                        name=f"wind_cap[k={k},t={t},s={s}]",
                    )
                else:
                    m.addConstr(
                        y[k, t, s] <= x[k] * alpha[k],
                        name=f"cap_use[k={k},t={t},s={s}]",
                    )

    # Tech-specific capacity upper bounds
    for k in techs:
        m.addConstr(x[k] <= x_ub[k], name=f"x_ub[{k}]")

    # Global capacity cap (optional)
    if X_max is not None:
        m.addConstr(
            gp.quicksum(x[k] for k in techs) <= X_max,
            name="X_max",
        )

    # Expected operating profit
    exp_operating_profit = gp.LinExpr()
    for s in scens:
        for k in techs:
            for t in hours:
                exp_operating_profit += (
                    p_s[s] * (lambdas_scen[s, t] - c[k, t]) * y[k, t, s]
                )

    # CAPEX cost
    capex_cost = gp.quicksum(C_capex_per_MW[k] * x[k] for k in techs)

    # Objective: max expected profit minus CAPEX
    m.setObjective(exp_operating_profit - capex_cost, GRB.MAXIMIZE)

    m.optimize()

    x_sol = np.zeros(K)
    y_sol = np.zeros((S, K, T))

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        for k in techs:
            x_sol[k] = x[k].X
        for s in scens:
            for t in hours:
                for k in techs:
                    y_sol[s, k, t] = y[k, t, s].X

        print(f"\nModel 4 (uncertainty) – Optimal expected objective value: {m.ObjVal:.2f}")
        total_capex = float(np.sum(C_capex_per_MW * x_sol))
        print(f"Total CAPEX investment: {total_capex:.2f} €\n")
        print("Optimal capacities x_k [MW]:")
        for name, val, capex in zip(tech_names, x_sol, C_capex_per_MW):
            print(f"  {name:8s}: {val:7.2f} MW  (CAPEX per MW = {capex:7.1f} €/MW)")
    else:
        print("Model did not solve to optimality. Status:", m.Status)

    return m, x_sol, y_sol


# ---------------------------------------------------------------------------
# Main: build nominal data, add uncertainty, generate scenarios, plot results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ========================
    # 0) BASIC SETUP (same as Model 3)
    # ========================
    T = 24
    hours = np.arange(T)

    tech_names = ["wind", "coal", "oil", "biomass", "nuclear"]
    K = len(tech_names)

    # Demand profile (nominal)
    base_load = 500.0
    morning_peak = 150.0 * np.exp(-0.5 * ((hours - 8.0) / 3.0) ** 2)
    evening_peak = 250.0 * np.exp(-0.5 * ((hours - 19.0) / 3.0) ** 2)
    D_base = (base_load + morning_peak + evening_peak).round(1)

    # Price profile (nominal)
    lambda_base = 20.0 + 0.09 * (D_base - D_base.min())
    lambda_base += np.random.normal(0.0, 3.0, T)
    lambda_base = np.clip(lambda_base, 20.0, 130.0).round(2)

    # Marginal costs (same as Model 3, with tiny noise)
    c_base = np.array([10.0, 45.0, 100.0, 60.0, 12.0])
    c = np.tile(c_base.reshape(K, 1), (1, T))
    c += np.random.normal(0.0, 0.5, (K, T))
    c = np.clip(c, 0.0, None).round(2)

    # Availability factors
    alpha = np.array([0.50, 1.0, 1.0, 1.0, 1.0])

    # Base CAPEX per MW
    base_capex = 300.0
    base_capex_per_mw = np.array([
        0.6 * base_capex,  # wind
        0.4 * base_capex,  # coal
        0.8 * base_capex,  # oil
        0.5 * base_capex,  # biomass
        1.0 * base_capex,  # nuclear
    ])

    # Auction for CAPEX
    n_buyers_run = 19
    C_capex_per_MW = simulate_capex_auction(
        base_capex_per_mw=base_capex_per_mw,
        n_buyers=n_buyers_run,
        competition_intensity=4.0,
        saturation_scale=5.0,
        noise_level=0.05,
        seed=42,
    )

    print("\nAuction-derived CAPEX per MW [€/MW]:")
    for name, cap in zip(tech_names, C_capex_per_MW):
        print(f"  {name:8s}: {cap:7.1f} €/MW")

    # Capacity bounds and global cap
    x_ub = np.array([400.0, 400.0, 300.0, 300.0, 500.0], dtype=float)
    X_max = 900.0

    # Wind CF (nominal pattern)
    CF_wind_base = 0.4 + 0.2 * np.sin(2.0 * np.pi * (hours - 3.0) / 24.0)
    CF_wind_base = np.clip(CF_wind_base, 0.05, 0.8)

    # ========================
    # 1) SCENARIOS FROM ERROR RANGES (NOW STRONGER / CLEARER)
    # ========================

    S = 3
    p_s = np.array([0.3, 0.4, 0.3])  # you can keep this

    lambdas_scen = np.zeros((S, T))
    D_scen       = np.zeros((S, T))
    CF_wind_scen = np.zeros((S, T))

    # Scenario 0: BASE CASE
    lambdas_scen[0, :] = lambda_base
    D_scen[0, :]       = D_base
    CF_wind_scen[0, :] = CF_wind_base

    # Scenario 1: "STRESS" – high demand, high price, LOW wind
    D_scen[1, :]       = 1.20 * D_base          # +20% demand
    lambdas_scen[1, :] = 1.25 * lambda_base     # +25% prices
    CF_wind_scen[1, :] = np.clip(0.6 * CF_wind_base, 0.0, 1.0)  # 40% less wind

    # Scenario 2: "GREEN & CHEAP" – lower demand, lower price, HIGH wind
    D_scen[2, :]       = 0.85 * D_base          # -15% demand
    lambdas_scen[2, :] = 0.80 * lambda_base     # -20% prices
    CF_wind_scen[2, :] = np.clip(1.4 * CF_wind_base, 0.0, 1.0)  # 40% more wind

    # (optional) round for nicer printing
    lambdas_scen = lambdas_scen.round(2)
    D_scen       = D_scen.round(1)
    CF_wind_scen = CF_wind_scen.round(3)

    # ========================
    # 2) SOLVE STOCHASTIC MODEL
    # ========================

    model4, x_opt4, y_opt4 = solve_model4(
        lambdas_scen=lambdas_scen,
        c=c,
        D_scen=D_scen,
        CF_wind_scen=CF_wind_scen,
        C_capex_per_MW=C_capex_per_MW,
        x_ub=x_ub,
        p_s=p_s,
        alpha=alpha,
        X_max=X_max,
        tech_names=tech_names,
    )

    # Note: x_opt4 is the SAME for all scenarios (here-and-now capacity decision).
    # Uncertainty shows up in dispatch y_opt4[s, k, t] and in profits.

    S, K, T = y_opt4.shape
    total_capex = float(np.sum(C_capex_per_MW * x_opt4))
    print(f"\nTotal CAPEX investment: {total_capex:.2f} €")

    # ========================
    # 3) SCENARIO-WISE REVENUE, COST, PROFIT
    # ========================

    scenario_revenue  = np.zeros(S)
    scenario_var_cost = np.zeros(S)
    scenario_profit   = np.zeros(S)

    for s in range(S):
        # Revenue = sum_t λ_{s,t} * D_{s,t} (demand is fully served in every scenario)
        scenario_revenue[s] = np.sum(lambdas_scen[s, :] * D_scen[s, :])

        # Variable cost = sum_{k,t} c_{k,t} * y_{s,k,t}
        for k in range(K):
            scenario_var_cost[s] += np.sum(c[k, :] * y_opt4[s, k, :])

        scenario_profit[s] = scenario_revenue[s] - scenario_var_cost[s] - total_capex

        print(f"\nScenario {s}:")
        print(f"  Revenue      = {scenario_revenue[s]:10.2f} €")
        print(f"  Variable cost= {scenario_var_cost[s]:10.2f} €")
        print(f"  Profit       = {scenario_profit[s]:10.2f} €")

    # ---- Plot 1: scenario-wise revenue, cost, profit ----
    plt.figure(figsize=(8, 4))
    x_pos = np.arange(S)

    plt.bar(x_pos - 0.25, scenario_revenue,  width=0.25, label="Revenue")
    plt.bar(x_pos,         scenario_var_cost, width=0.25, label="Variable cost")
    plt.bar(x_pos + 0.25, scenario_profit,    width=0.25, label="Profit")

    plt.xticks(x_pos, [f"Base", "Stress", "Green"])
    plt.xlabel("Scenario")
    plt.ylabel("€ over 24 h")
    plt.title("Model 4 – Impact of uncertainty on economics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================
    # 4) TECHNOLOGY MIX PER SCENARIO
    # ========================

    total_gen_scen = np.sum(y_opt4, axis=2)  # (S, K)
    shares = total_gen_scen / total_gen_scen.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 4))
    bottom = np.zeros(S)

    for k in range(K):
        plt.bar(
            x_pos,
            shares[:, k],
            bottom=bottom,
            label=tech_names[k],
        )
        bottom += shares[:, k]

    plt.xticks(x_pos, ["Base", "Stress", "Green"])
    plt.ylabel("Share of total generation [-]")
    plt.title("Model 4 – Technology mix under different futures")
    plt.legend(loc="upper right", ncols=2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================
    # 5) WIND UTILISATION VS CF
    # ========================

    idx_wind = tech_names.index("wind")
    plt.figure(figsize=(8, 4))

    for s, label in enumerate(["Base", "Stress", "Green"]):
        if x_opt4[idx_wind] > 1e-6:
            wind_util = y_opt4[s, idx_wind, :] / x_opt4[idx_wind]
            plt.plot(
                hours,
                wind_util,
                linestyle="--",
                label=f"Utilisation {label} (p={p_s[s]:.2f})",
            )

    plt.plot(hours, CF_wind_base, linewidth=2, label="Nominal wind CF")

    plt.xlabel("Hour of day")
    plt.ylabel("Fraction of capacity")
    plt.ylim(0.0, 1.05)
    plt.title("Model 4 – Wind utilisation across scenarios")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ========================
    # 6) EXPECTED CONTRIBUTION VS CAPEX (to see "optimal CAPEX")
    # ========================

    # Expected generation per technology over all scenarios
    exp_gen_k = np.zeros(K)
    for k in range(K):
        for s in range(S):
            exp_gen_k[k] += p_s[s] * np.sum(y_opt4[s, k, :])  # [MWh over 24 h]

    # Expected operating margin per tech (revenue - var cost, excluding CAPEX)
    exp_margin_k = np.zeros(K)
    for k in range(K):
        for s in range(S):
            # scenario-specific margin: sum_t (λ_{s,t} - c_{k,t}) * y_{s,k,t}
            margin_s = 0.0
            for t in range(T):
                margin_s += (lambdas_scen[s, t] - c[k, t]) * y_opt4[s, k, t]
            exp_margin_k[k] += p_s[s] * margin_s

    # Plot: expected margin vs CAPEX per tech (to see if CAPEX is "worth it")
    plt.figure(figsize=(7, 5))
    for k, name in enumerate(tech_names):
        plt.scatter(C_capex_per_MW[k], exp_margin_k[k], s=60)
        plt.text(C_capex_per_MW[k] * 1.01, exp_margin_k[k] * 1.01, name)

    plt.xlabel("CAPEX per MW [€/MW]")
    plt.ylabel("Expected 24 h operating margin [€]")
    plt.title("Model 4 – Expected margin vs CAPEX (by technology)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    