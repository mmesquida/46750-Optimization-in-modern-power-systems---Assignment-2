import numpy as np
import gurobipy as gp
from gurobipy import GRB

"""
Model_4.py – Stochastic upgrade of Model 3

Two-stage stochastic investment + dispatch model:

Stage 1 (here-and-now):
    - Choose capacities x[k] [MW] for each technology k (same across all scenarios)

Stage 2 (recourse, scenario-dependent):
    - For each scenario s and hour t, choose dispatch y[k,t,s] [MWh]

Uncertainty:
    - Hourly demand D_scen[s,t]
    - Hourly prices lambdas_scen[s,t]
    - Wind capacity factor CF_wind_scen[s,t]

Demand must always be met in every scenario and hour:
    sum_k y[k,t,s] = D_scen[s,t]

Objective:
    Maximise expected profit across scenarios:
        E_s[ sum_{t,k} (λ_{s,t} - c_{k,t}) y_{k,t,s} ] - sum_k C_capex_k * x_k

CAPEX C_capex_k comes from the same auction simulation used in Model 3.
"""


gp.setParam("LogToConsole", 0)

# Auction function from Model 3 (for CAPEX per MW)

def simulate_capex_auction(
    base_capex_per_mw,
    n_units=10,          # number of generator units available per technology
    n_bidders=15,        # number of buyers (including us)
    seed=42,
):
    """
    Forward auction for generators:
    - Seller offers n_units generators of each technology.
    - Each generator has a cost around base_capex_per_mw.
    - n_bidders buyers each want 1 generator.
    - Clearing CAPEX = cost of the marginal unit that satisfies total demand.
      More bidders -> higher demand -> higher clearing CAPEX.
    """
    np.random.seed(seed)
    base_capex_per_mw = np.asarray(base_capex_per_mw, dtype=float)
    K = len(base_capex_per_mw)
    C_capex = np.zeros(K)

    demand_units = max(1, n_bidders)

    for k in range(K):
        supply_costs = base_capex_per_mw[k] * np.random.uniform(0.8, 1.2, size=n_units)
        supply_costs = np.sort(supply_costs)

        idx = min(demand_units - 1, n_units - 1)
        C_capex[k] = supply_costs[idx]

    return C_capex

# Model 4 – two-stage stochastic model

def solve_model4(
    lambdas_scen,       # shape (S,T): λ_{s,t}
    c,                  # shape (K,T): c_{k,t} (same across scenarios)
    D_scen,             # shape (S,T): D_{s,t}
    CF_wind_scen,       # shape (S,T): CF_wind_{s,t}
    C_capex_per_MW,     # shape (K,): CAPEX per MW (from auction / Model 3)
    x_ub,               # shape (K,): tech-specific capacity upper bounds
    p_s,                # shape (S,): scenario probabilities (sum to 1)
    X_max=None,         # optional: global cap on sum_k x_k
    tech_names=None,
):
    
    lambdas_scen = np.asarray(lambdas_scen, dtype=float)   # (S,T)
    D_scen       = np.asarray(D_scen, dtype=float)         # (S,T)
    CF_wind_scen = np.asarray(CF_wind_scen, dtype=float)   # (S,T)
    c            = np.asarray(c, dtype=float)              # (K,T)
    C_capex_per_MW = np.asarray(C_capex_per_MW, dtype=float)  # (K,)
    x_ub         = np.asarray(x_ub, dtype=float)           # (K,)
    p_s          = np.asarray(p_s, dtype=float)            # (S,)

    S, T = lambdas_scen.shape
    assert D_scen.shape == (S, T)
    assert CF_wind_scen.shape == (S, T)
    K, T_c = c.shape
    assert T_c == T
    assert C_capex_per_MW.shape == (K,)
    assert x_ub.shape == (K,)
    assert p_s.shape == (S,)
    assert np.isclose(p_s.sum(), 1.0), "Scenario probabilities must sum to 1."

    if tech_names is None:
        tech_names = [f"tech_{k}" for k in range(K)]
    else:
        assert len(tech_names) == K

    hours = range(T)
    techs = range(K)
    scens = range(S)

    # Find wind index
    lower_names = [name.lower() for name in tech_names]
    idx_wind = lower_names.index("wind")

    m = gp.Model("Model4_stochastic")

    # Stage 1: capacities x[k] [MW]
    x = m.addVars(techs, lb=0.0, name="x")

    # Stage 2: dispatch y[k,t,s] [MWh]
    y = m.addVars(techs, hours, scens, lb=0.0, name="y")

    # Demand balance: sum_k y[k,t,s] = D_scen[s,t]
    for s in scens:
        for t in hours:
            m.addConstr(
                gp.quicksum(y[k, t, s] for k in techs) == D_scen[s, t],
                name=f"demand_balance[s={s},t={t}]",
            )

    # Capacity constraints for dispatchables: y[k,t,s] <= x[k]
    for k in techs:
        if k == idx_wind:
            continue  # wind handled separately below
        for s in scens:
            for t in hours:
                m.addConstr(
                    y[k, t, s] <= x[k],
                    name=f"cap_use[k={k},t={t},s={s}]",
                )

    # Wind availability: y[wind,t,s] <= x[wind] * CF_wind_scen[s,t]
    for s in scens:
        for t in hours:
            m.addConstr(
                y[idx_wind, t, s] <= x[idx_wind] * CF_wind_scen[s, t],
                name=f"wind_avail[t={t},s={s}]",
            )

    # Tech-specific capacity bounds: x[k] <= x_ub[k]
    for k in techs:
        m.addConstr(x[k] <= x_ub[k], name=f"x_ub[{k}]")

    # Optional global capacity cap
    if X_max is not None:
        m.addConstr(
            gp.quicksum(x[k] for k in techs) <= X_max,
            name="X_max",
        )

    # Expected operating profit: Σ_s p_s Σ_{t,k} (λ_{s,t} - c_{k,t}) y[k,t,s]
    exp_operating_profit = gp.LinExpr()
    for s in scens:
        for k in techs:
            for t in hours:
                exp_operating_profit += (
                    p_s[s] * (lambdas_scen[s, t] - c[k, t]) * y[k, t, s]
                )

    # CAPEX cost: Σ_k C_capex_k * x[k]
    capex_cost = gp.quicksum(C_capex_per_MW[k] * x[k] for k in techs)

    # Maximise expected profit
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

        print(f"\nModel 4 – Optimal expected objective value: {m.ObjVal:.2f}\n")
        print("Optimal capacities x_k [MW]:")
        for k, name in enumerate(tech_names):
            print(f"  {name}: {x_sol[k]:.2f} MW")

        for s in scens:
            print(f"\nScenario {s} (p = {p_s[s]:.2f}):")
            for k, name in enumerate(tech_names):
                gen_k = y_sol[s, k, :].sum()
                print(f"  {name}: {gen_k:.2f} MWh")

    else:
        print("Model did not solve to optimality. Status:", m.Status)

    return m, x_sol, y_sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 24
    K = 5
    hours = np.arange(T)

    tech_names = ["Nuclear", "Biomass", "Gas", "Wind", "Oil"]

    # Demand profile
    base_load = 500
    morning_peak = 150 * np.exp(-0.5 * ((hours - 8) / 3) ** 2)
    evening_peak = 250 * np.exp(-0.5 * ((hours - 19) / 3) ** 2)
    D_base = base_load + morning_peak + evening_peak
    D_base = D_base.round(1)

    # Price profile
    lambda_base = 20 + 0.09 * (D_base - D_base.min())
    lambda_base += np.random.normal(0, 3, T)
    lambda_base = np.clip(lambda_base, 20, 130)
    lambda_base = lambda_base.round(2)

    # Marginal costs c_{k,t}
    c_base = np.array([10.0, 30.0, 60.0, 2.0, 40.0])
    c = np.tile(c_base.reshape(K, 1), (1, T))
    c += np.random.normal(0, 0.5, (K, T))
    c = np.clip(c, 0, None)
    c = c.round(2)

    # Base CAPEX per MW (these are *base* costs used in the auction)

    base_capex = 260 # €/MW
    base_capex_per_mw = np.array([0.6 * base_capex, # wind cheaper relative to its margin
                                  0.4 * base_capex, # coal cheap
                                  1.0 * base_capex, # oil insanely expensive -> never used
                                  0.5 * base_capex, # biomass cheap
                                  1 * base_capex])  # nuclear expensive relative to its margin


    # Run auction to get CAPEX per MW
    C_capex_per_MW = simulate_capex_auction(
        base_capex_per_mw=base_capex_per_mw,
        n_units=10,
        n_bidders=12,
        seed=42,
    )

    print("Auction-derived CAPEX per MW [€/MW]:")
    for name, cap in zip(tech_names, C_capex_per_MW):
        print(f"  {name}: {cap:.1f} €/MW")

    # Capacity upper bounds and global cap
    x_ub = np.array([500, 300, 300, 400, 350], dtype=float)
    X_max = 1200.0

    # Base wind capacity factor
    CF_wind_base = 0.4 + 0.2 * np.sin(2 * np.pi * (hours - 3) / 24)
    CF_wind_base = np.clip(CF_wind_base, 0.05, 0.8)

    # Built 3 scenarios
    S = 3
    p_s = np.array([0.3, 0.4, 0.3])

    lambdas_scen = np.zeros((S, T))
    D_scen = np.zeros((S, T))
    CF_wind_scen = np.zeros((S, T))

    # Scenario 0: base
    lambdas_scen[0, :] = lambda_base
    D_scen[0, :] = D_base
    CF_wind_scen[0, :] = CF_wind_base

    # Scenario 1: high wind, low demand
    lambdas_scen[1, :] = lambda_base
    D_scen[1, :] = 0.9 * D_base
    CF_wind_scen[1, :] = np.clip(CF_wind_base * 1.3, 0, 1)

    # Scenario 2: low wind, high demand
    lambdas_scen[2, :] = lambda_base
    D_scen[2, :] = 1.1 * D_base
    CF_wind_scen[2, :] = np.clip(CF_wind_base * 0.7, 0, 1)

    # Solve model 4
    model4, x_opt4, y_opt4 = solve_model4(
        lambdas_scen=lambdas_scen,
        c=c,
        D_scen=D_scen,
        CF_wind_scen=CF_wind_scen,
        C_capex_per_MW=C_capex_per_MW,
        x_ub=x_ub,
        p_s=p_s,
        X_max=X_max,
        tech_names=tech_names,
    )

S, K, T = y_opt4.shape

# Total CAPEX (same for all scenarios)
capex_total = float(np.sum(C_capex_per_MW * x_opt4))

scenario_revenue = np.zeros(S)
scenario_var_cost = np.zeros(S)
scenario_profit = np.zeros(S)

for s in range(S):
    # revenue: sum_t λ_{s,t} * D_{s,t}
    scenario_revenue[s] = np.sum(lambdas_scen[s, :] * D_scen[s, :])

    # variable cost: sum_{k,t} c_{k,t} * y_{s,k,t}
    for k in range(K):
        scenario_var_cost[s] += np.sum(c[k, :] * y_opt4[s, k, :])

    # profit per scenario (CAPEX subtracted once in each scenario for comparison)
    scenario_profit[s] = scenario_revenue[s] - scenario_var_cost[s] - capex_total

# --- Plot ---
plt.figure(figsize=(8, 4))
x = np.arange(S)

plt.bar(x - 0.25, scenario_revenue, width=0.25, label="Revenue")
plt.bar(x,         scenario_var_cost, width=0.25, label="Variable cost")
plt.bar(x + 0.25, scenario_profit, width=0.25, label="Profit")

plt.xticks(x, [f"Scen {s}" for s in range(S)])
plt.xlabel("Scenario")
plt.ylabel("€ over 24 h")
plt.title("Model 4 – scenario-wise revenue, cost and profit")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

total_gen_scen = np.sum(y_opt4, axis=2)  # shape (S, K): total MWh per scen, tech

S, K = total_gen_scen.shape
shares = total_gen_scen / total_gen_scen.sum(axis=1, keepdims=True)  # fractions

plt.figure(figsize=(8, 4))
x = np.arange(S)

bottom = np.zeros(S)
for k in range(K):
    plt.bar(
        x,
        shares[:, k],
        bottom=bottom,
        label=tech_names[k]
    )
    bottom += shares[:, k]

plt.xticks(x, [f"Scen {s}" for s in range(S)])
plt.ylabel("Share of total generation [-]")
plt.title("Model 4 – Technology share in each scenario")
plt.legend(loc="upper right", ncols=2)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

idx_wind = tech_names.index("Wind")
hours = np.arange(T)

plt.figure(figsize=(8, 4))

for s in range(S):
    if x_opt4[idx_wind] > 1e-6:
        wind_util = y_opt4[s, idx_wind, :] / x_opt4[idx_wind]
        plt.plot(hours, wind_util, linestyle="--",
                 label=f"Utilisation scen {s} (p={p_s[s]:.2f})")

# Plot CF of base scenario (or all)
plt.plot(hours, CF_wind_scen[0, :], linewidth=2, label="Wind CF scen 0")

plt.xlabel("Hour of day")
plt.ylabel("Fraction of capacity")
plt.ylim(0, 1.05)
plt.title("Model 4 – wind utilisation vs forecast")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()




