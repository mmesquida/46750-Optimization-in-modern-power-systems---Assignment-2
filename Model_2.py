"""
Model 2 — Joint Investment and Dispatch with Capacity Cap Sensitivity

This script implements Model 2 from the assignment: a 24-hour investment
and dispatch optimisation model where the planner chooses the installed
capacity of each technology and its hourly generation profile. A linear
capacity expansion model is combined with a realistic 24-hour market
environment (demand, prices, and wind availability), and the impact of
allowing progressively more overinvestment is analysed via a series of
capacity-cap scenarios.

The script:
1. Builds a synthetic 24-hour market environment.
2. Solves the base-case optimisation with a system-wide capacity cap
   equal to the peak demand (X_max = 500 MW).
3. Generates a range of scenarios where the total capacity cap X_max is
   increased from 0% up to 50% above the peak demand.
4. For each scenario, solves the investment–dispatch problem using Gurobi,
   storing optimal capacities, generation and profit.
5. Produces:
   - a plot of profit vs. allowed total capacity with a stacked bar
     decomposition of the installed capacity by technology,
   - a stacked area plot of hourly generation by technology compared
     against demand.

Usage:
------
All outputs are produced automatically by running the script; no command-line arguments are required.

Key parameters to experiment with:
----------------------------------
- Capex_one_day     : dailyised CAPEX per MW of installed capacity,
                      which controls how attractive overinvestment is.
- x_ub              : technology-specific capacity upper bounds
                      (default 200 MW per technology).
- X_max             : base-case system-wide capacity cap (500 MW before
                      scaling by scenario_scales).
- scenario_scales   : list of multipliers applied to the peak demand to
                      define X_max in each scenario (e.g. 1.0 to 1.5).
- c_base and noise  : baseline marginal costs and their random perturbation.
- CF_wind profile   : shape and randomness of the wind capacity factor,
                      which affect the value of investing in wind capacity.

These parameters allow exploration of how the allowed total capacity,
technology cost structure and wind availability shape the optimal
capacity mix, the hourly dispatch pattern, and the resulting profit.
"""


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt


def solve_model2(lambdas, c, D,
                 C_capex_per_MW,
                 x_ub,
                 CF_wind,
                 X_max=None,
                 tech_names=None,
                 min_wind_capacity=None):
    
    lambdas = np.asarray(lambdas, dtype=float)
    c       = np.asarray(c, dtype=float)
    D       = np.asarray(D, dtype=float)
    C_capex_per_MW = np.asarray(C_capex_per_MW, dtype=float)
    x_ub    = np.asarray(x_ub, dtype=float)
    CF_wind = np.asarray(CF_wind, dtype=float)

    K = c.shape[0]          
    T = lambdas.shape[0]    
    if tech_names is None:
        tech_names = [f"tech_{k}" for k in range(K)]
    m = gp.Model("Model2_invest_dispatch")

    hours = range(T)   # i
    techs = range(K)   # k

    
    idx_wind = tech_names.index("Wind")

    # Decision variables
    # x[k] : capacity (MW) of tech k (investment decision)
    x = m.addVars(techs, name="x", lb=0.0)

    # y[k,i] : generation (MWh) of tech k in hour i
    y = m.addVars(techs, hours, name="y", lb=0.0)



    # Objective: max sum_i,k (λ_i - c_{k,i}) y_{k,i} - sum_k C_capex_k * x_k
    profit_expr = gp.quicksum(
        (lambdas[i] - c[k, i]) * y[k, i]
        for k in techs
        for i in hours
    ) - gp.quicksum(C_capex_per_MW * x[k] for k in techs) 

    m.setObjective(profit_expr, GRB.MAXIMIZE)

    # 1) Hourly demand balance
    for i in hours:
        m.addConstr(
            gp.quicksum(y[k, i] for k in techs) == D[i],
            name=f"demand_balance[{i}]"
        )


    # Capacity constraint for dispatchable techs
    for k in techs:
        if k == idx_wind:
            continue   
        for i in hours:
            m.addConstr(
                y[k, i] <= x[k],
                name=f"cap_use[{k},{i}]"
            )

    # Wind availability:
    for i in hours:
        m.addConstr(
            y[idx_wind, i] <= x[idx_wind] * CF_wind[i],
            name=f"wind_availability[{i}]"
        )

    # 3) Technology-specific upper bounds on total capacity
    for k in techs:
        m.addConstr(
            x[k] <= x_ub[k],
            name=f"x_ub[{k}]"
        )

    # 4)  Global capacity cap
    if X_max is not None:
        m.addConstr(
            gp.quicksum(x[k] for k in techs) <= X_max,
            name="X_max"
        )

    # Optimize
    m.optimize()
    x_sol = np.zeros(K)
    y_sol = np.zeros((K, T))

    if m.Status == GRB.OPTIMAL:
        print(f"\nOptimal objective value (profit): {m.ObjVal:.2f}\n")

        for k in techs:
            x_sol[k] = x[k].X

        print("Optimal capacities x_k [MW]:")
        for k in techs:
            print(f"  {tech_names[k]}: {x_sol[k]:.2f} MW")

        print("\nTotal generation per technology over 24h:")
        for k in techs:
            gen_k = 0.0
            for i in hours:
                y_sol[k, i] = y[k, i].X
                gen_k += y_sol[k, i]
            print(f"  {tech_names[k]}: {gen_k:.2f} MWh")

    else:
        print("Model did not solve to optimality. Status:", m.Status)

    return m, x_sol, y_sol

if __name__ == "__main__":
    T = 24
    K = 5
    hours = np.arange(T)
    
    # 1) Demand D profile
    base_load = 300
    morning_peak = 150 * np.exp(-0.5 * ((hours - 8) / 3)**2)
    evening_peak = 250 * np.exp(-0.5 * ((hours - 19) / 3)**2)

    D = base_load + morning_peak + evening_peak
    factor = 500 / D.max()
    D = (D * factor).round(1)

    # 2) Prices λ_i
    lambda_i = 60 + 0.09 * (D - D.min())
    lambda_i += np.random.normal(0, 3, T)
    lambda_i = np.clip(lambda_i, 20, 130)
    lambda_i = lambda_i.round(2)

    # 3) Marginal costs 
    c_base = np.array([10.0, 45.0, 100.0, 60.0, 12.0])
    c = np.tile(c_base.reshape(K, 1), (1, T))
    c += np.random.normal(0, 0.5, (K, T))
    c = np.clip(c, 0, None)
    c = c.round(2)

    tech_names = ["Wind", "Coal", "Oil", "Biomass", "Nuclear"]

    # 4) CAPEX per MW 
    Capex_one_day = 1000000000 / (25 * 365.25*500) # EUR per MW for 1 day 

    # 5) Max allowed capacity per tech (upper bounds)
    x_ub = np.array([200, 200, 200, 200, 200], dtype=float)

    # 6) Optional global cap
    X_max = 500.0  

    # 7) Wind forecast
    rng = np.random.default_rng(seed=123)

    CF_wind = 0.7 + 0.37 * np.sin(2 * np.pi * (hours - 3) / 24)
    CF_wind += rng.normal(0.0, 0.10, size=hours.shape[0])
    CF_wind = np.clip(CF_wind, 0.0, 1.0)
    print("CF_wind min, max =", CF_wind.min(), CF_wind.max())

    model_base, x_opt_base, y_opt_base = solve_model2(
        lambdas=lambda_i,
        c=c,
        D=D,
        C_capex_per_MW=Capex_one_day,
        x_ub=x_ub,
        CF_wind=CF_wind,
        X_max=X_max,           
        tech_names=tech_names
    )

    idx_wind = tech_names.index("Wind")

    D_max = D.max()

    scenario_scales = [1.0, 1.05, 1.1, 1.15, 1.20,1.25, 1.3,1.35,1.4,1.45,1.5]  
    overinvestment_pct = [(s - 1.0) * 100 for s in scenario_scales]

    profits = []
    x_solutions = []
    y_solutions = []  

    # Scenario generator: Allow different total capacity caps

    for scale in scenario_scales:
        X_max_scen = scale * D_max   

        print("\n==============================")
        print(f"Scenario: X_max = {X_max_scen:.1f} MW "
            f"({(scale-1)*100:.0f}% extra over peak)")
        print("==============================")

        model2, x_opt, y_opt = solve_model2(
            lambdas=lambda_i,
            c=c,
            D=D,
            C_capex_per_MW=Capex_one_day,
            x_ub=x_ub,
            CF_wind=CF_wind,
            X_max=X_max_scen,
            tech_names=tech_names,
            min_wind_capacity=None
        )

        profits.append(model2.ObjVal)
        x_solutions.append(x_opt)

    x_solutions = np.vstack(x_solutions).T
    profits = np.array(profits)

phi = (1+np.sqrt(5))/2  
x_length = 10
y_length = x_length / phi
plt.figure(figsize=(x_length, y_length))    
fig, ax1 = plt.subplots(figsize=(x_length, y_length))


ax1.plot(overinvestment_pct, profits, marker="o", linewidth=2, label="Profit")

y_min = min(profits)
y_max = max(profits)
rango = y_max - y_min
ax1.set_ylim(y_min - 0.05*rango, y_max + 0.25*rango)

ax1.set_xlabel("Allowed total capacity [% over peak demand]")
ax1.set_ylabel("Profit over 24 h [€]")

ax2 = ax1.twinx()

n_scen = len(overinvestment_pct)
bottom = np.zeros(n_scen)
bar_width = 4.0  

tech_colors = [] 

for k, tech in enumerate(tech_names):
    bars_k = ax2.bar(
        overinvestment_pct,
        x_solutions[k, :],
        width=bar_width,
        bottom=bottom,
        alpha=0.35,
        label=tech,
    )
    bottom = bottom + x_solutions[k, :]
    tech_colors.append(bars_k[0].get_facecolor())

ax2.set_ylabel("Installed capacity [MW]")

lines, labels = ax1.get_legend_handles_labels()
bars, bar_labels = ax2.get_legend_handles_labels()

ax1.legend(
    lines + bars,
    labels + bar_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20), 
    ncols=3,
    frameon=False
)

plt.title("Profit vs total capacity capacity and technology mix", fontsize=14)
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# --- Stacked generation plot ---
plt.figure(figsize=(x_length, y_length))

plt.stackplot(
    hours,
    *[y_opt[k, :] for k in range(len(tech_names))],
    labels=tech_names,
    step="post",
    colors=tech_colors,   
    alpha=0.4            
)

plt.plot(hours, D, linestyle="--", linewidth=2, label="Demand")
plt.xlabel("Hour of day")
plt.ylabel("Generation [MWh]")
plt.title("Hourly generation by technology vs demand")
plt.legend(loc="upper left", ncols=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




