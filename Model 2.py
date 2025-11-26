import gurobipy as gp
from gurobipy import GRB
import numpy as np

T = 24   # hours in a day
K = 5    # tecnolology types

hours = np.arange(T)

# 1) Hourly demand D_i (MWh)
#  Curve with valley at night and peak in the afternoon

base_load = 500  # base demand
morning_peak = 150 * np.exp(-0.5 * ((hours - 8) / 3)**2)
evening_peak = 250 * np.exp(-0.5 * ((hours - 19) / 3)**2)
D = base_load + morning_peak + evening_peak
D = D.round(1)  # opcional

# 2) Price lambda_i (€/MWh)
#  Correlated with demand + noise, bounded between 20 and 130 €/MWh

lambda_i = 20 + 0.09 * (D - D.min())  # Demand-correlated part
lambda_i += np.random.normal(0, 3, T) # noise
lambda_i = np.clip(lambda_i, 20, 130) # límits
lambda_i = lambda_i.round(2)


# 3) Marginal cost c_{k,i} (€/MWh)
#    Constant per technology + small noise
#    [Nuclear, Biomass, Gas, Wind, Solar]

c_base = np.array([10.0, 30.0, 60.0, 2.0, 1.0])  #average marginal costs
c = np.tile(c_base.reshape(K, 1), (1, T))
c += np.random.normal(0, 0.5, (K, T))  # noise
c = np.clip(c, 0, None)
c = c.round(2)

# 4) Capacities x_k (MW)

x_k = np.array([300, 150, 200, 250, 180], dtype=float)

print("Max demand:", D.max())
print("Total capacity:", x_k.sum())

# -------------------------------
# 5) CAPEX fix(€/day)
# -------------------------------
C_capex_fixed = 100_000.0

tech_names = ["Nuclear", "Biomass", "Gas", "Wind", "Solar"]

def solve_model2(lambdas, c, D, x_cap, C_capex_fixed, tech_names=None):
    # Convert inputs to numpy arrays for safety
    lambdas = np.asarray(lambdas)
    c       = np.asarray(c)
    D       = np.asarray(D)
    x_cap   = np.asarray(x_cap)

    K = c.shape[0]          # number of technologies
    T = lambdas.shape[0]    # number of hours (24)
    if tech_names is None:
        tech_names = [f"tech_{k}" for k in range(K)]

    # Create model
    m = gp.Model("Model2_dispatch")

    # Sets for convenience
    hours = range(T)   # i
    techs = range(K)   # k

    # Decision variables y_{k,i} >= 0  [MWh]
    y = m.addVars(techs, hours, name="y", lb=0.0)

    # Objective: max sum_i sum_k (λ_i - c_{k,i}) y_{k,i} - C_capex_fixed
    profit_expr = gp.quicksum(
        (lambdas[i] - c[k, i]) * y[k, i]
        for k in techs
        for i in hours
    ) - C_capex_fixed

    m.setObjective(profit_expr, GRB.MAXIMIZE)
    # 1) Hourly demand cap: sum_k y_{k,i} <= D_i  for all hours i
    for i in hours:
        m.addConstr(
            gp.quicksum(y[k, i] for k in techs) <= D[i],
            name=f"demand_cap[{i}]"
        )

    # 2) Capacity constraint: y_{k,i} <= x_k  for all k, i
    #    (you can multiply x_k by availability factor if needed)
    for k in techs:
        for i in hours:
            m.addConstr(
                y[k, i] <= x_cap[k],
                name=f"cap[{k},{i}]"
            )

    # Optimize
    m.optimize()

    # Extract solution
    y_sol = np.zeros((K, T))
    if m.status == GRB.OPTIMAL:
        for k in techs:
            for i in hours:
                y_sol[k, i] = y[k, i].X

        print(f"\nOptimal objective value (profit): {m.ObjVal:.2f}")
        for k in techs:
            print(f"Total generation of {tech_names[k]}: {y_sol[k, :].sum():.2f} MWh")

    else:
        print("Model did not solve to optimality. Status:", m.status)

    return m, y_sol

if __name__ == "__main__":
    model2, y_opt = solve_model2(
        lambdas=lambda_i,
        c=c,
        D=D,
        x_cap=x_k,
        C_capex_fixed=C_capex_fixed,
        tech_names=tech_names
    )