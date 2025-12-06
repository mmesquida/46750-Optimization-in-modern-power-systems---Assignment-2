import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt


def solve_model2(lambdas, c, D,
                 C_capex_per_MW,
                 x_ub,
                 CF_wind,
                 X_max=None,
                 tech_names=None):
    """
    Model 2: capacity investment + hourly dispatch (single day)

    - Decide capacities x[k] [MW] for each technology k
    - Decide hourly generation y[k,i] [MWh]
    - Objective: max profit over 24h minus CAPEX (per MW * capacity)

    Inputs
    ------
    lambdas : array shape (T,)       prices λ_i
    c       : array shape (K,T)      marginal costs c_{k,i}
    D       : array shape (T,)       hourly demand D_i
    C_capex_per_MW : array shape (K,)  CAPEX per MW for each technology
    x_ub    : array shape (K,)       max capacity per technology
    X_max   : scalar or None         optional total capacity cap
    tech_names : list of length K    names of technologies
    """

    # Convert to numpy arrays
    lambdas = np.asarray(lambdas, dtype=float)
    c       = np.asarray(c, dtype=float)
    D       = np.asarray(D, dtype=float)
    C_capex_per_MW = np.asarray(C_capex_per_MW, dtype=float)
    x_ub    = np.asarray(x_ub, dtype=float)

    K = c.shape[0]          # number of technologies
    T = lambdas.shape[0]    # number of hours
    if tech_names is None:
        tech_names = [f"tech_{k}" for k in range(K)]

    # Create model
    m = gp.Model("Model2_invest_dispatch")

    hours = range(T)   # i
    techs = range(K)   # k

    # Indices for wind
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
    ) - gp.quicksum(C_capex_per_MW[k] * x[k] for k in techs)

    m.setObjective(profit_expr, GRB.MAXIMIZE)

    # 1) Hourly demand balance: sum_k y_{k,i} == D_i
    for i in hours:
        m.addConstr(
            gp.quicksum(y[k, i] for k in techs) == D[i],
            name=f"demand_balance[{i}]"
        )


    # (2a) Capacity constraint for dispatchable techs: y[k,i] <= x[k]
    for k in techs:
        if k == idx_wind:
            continue   # handle wind separately
        for i in hours:
            m.addConstr(
                y[k, i] <= x[k],
                name=f"cap_use[{k},{i}]"
            )

    # (2b) Wind availability: y[wind,i] <= x[wind] * CF_wind[i]
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

    # 4) Optional global capacity cap
    if X_max is not None:
        m.addConstr(
            gp.quicksum(x[k] for k in techs) <= X_max,
            name="X_max"
        )

    # Optimize
    m.optimize()

    # Extract solution
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
    base_load = 500
    morning_peak = 150 * np.exp(-0.5 * ((hours - 8) / 3)**2)
    evening_peak = 250 * np.exp(-0.5 * ((hours - 19) / 3)**2)
    D = base_load + morning_peak + evening_peak
    D = D.round(1)

    # 2) Prices λ_i - > change 
    lambda_i = 20 + 0.09 * (D - D.min())
    lambda_i += np.random.normal(0, 3, T)
    lambda_i = np.clip(lambda_i, 20, 130)
    lambda_i = lambda_i.round(2)

    # 3) Marginal costs c_{k,i} 
    c_base = np.array([10.0, 30.0, 60.0, 2.0, 40.0])
    c = np.tile(c_base.reshape(K, 1), (1, T))
    c += np.random.normal(0, 0.5, (K, T))
    c = np.clip(c, 0, None)
    c = c.round(2)

    tech_names = ["Nuclear", "Biomass", "Gas", "Wind", "Oil"]

    # 4) CAPEX per MW 
    C_capex_per_MW = np.array([10, 20, 25, 5, 15], dtype=float)

    # 5) Max allowed capacity per tech (upper bounds)
    x_ub = np.array([500, 300, 300, 400, 350], dtype=float)

    # 6) Optional global cap
    X_max = 1200.0  # for example

    # 7) Wind forecast: capacity factor CF_wind[i] in [0,1]
    #   Slightly windier at night and early morning
    CF_wind = 0.4 + 0.2 * np.sin(2 * np.pi * (hours - 3) / 24)
    CF_wind = np.clip(CF_wind, 0.05, 0.8)  # avoid zero so model always has some wind


    model2, x_opt, y_opt = solve_model2(
        lambdas=lambda_i,
        c=c,
        D=D,
        C_capex_per_MW=C_capex_per_MW,
        x_ub=x_ub,
        CF_wind=CF_wind,
        X_max=X_max,
        tech_names=tech_names
    )


# --- Stacked generation plot ---
plt.figure(figsize=(8, 4))
plt.stackplot(hours,
              *[y_opt[k, :] for k in range(len(tech_names))],
              labels=tech_names,
              step="post")

plt.plot(hours, D, linestyle="--", linewidth=2, label="Demand")
plt.xlabel("Hour of day")
plt.ylabel("Generation [MWh]")
plt.title("Hourly generation by technology (stacked) vs demand")
plt.legend(loc="upper left", ncols=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Optimal capacity bar plot ---

plt.figure(figsize=(6, 4))

x_positions = np.arange(len(tech_names))
plt.bar(x_positions, x_opt)

plt.xticks(x_positions, tech_names, rotation=20)
plt.ylabel("Installed capacity [MW]")
plt.title("Optimal capacity mix (Model 2)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# --- Wind forecast vs utilisation plot ---

idx_wind = tech_names.index("Wind")

plt.figure(figsize=(8, 4))

plt.plot(hours, CF_wind, linewidth=2, label="Wind forecast CF (availability)")

if x_opt[idx_wind] > 1e-6:
    wind_util = y_opt[idx_wind, :] / x_opt[idx_wind]
    plt.plot(hours, wind_util, linestyle="--", linewidth=2,
             label="Wind utilisation (y_wind / x_wind)")
else:
    print("No wind capacity built (x_opt[Wind] = 0), skipping utilisation curve.")
    wind_util = None 

plt.xlabel("Hour of day")
plt.ylabel("Fraction of capacity")
plt.ylim(0, 1.05)
plt.title("Wind forecast vs actual utilisation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



