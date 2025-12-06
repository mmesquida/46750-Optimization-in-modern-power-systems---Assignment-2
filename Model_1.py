from typing import List, Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class Expando(object):
    pass


class InputDataModel1SingleHour:
    """
    Input data for single hour Model 1.

    lambda_price : scalar marginal revenue λ (€/MWh) in this hour
    c            : 1D array length n_tech with marginal costs c_k (€/MWh)
    alpha        : 1D array length n_tech with availability factors α_k (0..1)
                   e.g. α_wind, α_solar < 1; others = 1
    X_max        : total capacity budget (MW)
    C_capex_fixed: fixed capex term (can be zero if only argmax matters)
    """
    def __init__(
        self,
        lambda_price: float,
        c: np.ndarray,
        alpha: np.ndarray,
        X_max: float,
        C_capex_fixed: float = 0.0,
        tech_names: Optional[List[str]] = None,
    ):
        assert c.ndim == 1, "c must be 1D (length n_tech)"
        assert alpha.ndim == 1, "alpha must be 1D (length n_tech)"
        assert len(c) == len(alpha), "c and alpha must have same length"

        self.lambda_price = float(lambda_price)
        self.c = c.astype(float)
        self.alpha = alpha.astype(float)
        self.n_tech = len(c)
        self.tech = list(range(self.n_tech))
        self.tech_names = tech_names or [f"tech{k}" for k in range(self.n_tech)]
        self.X_max = float(X_max)
        self.C_capex_fixed = float(C_capex_fixed)


class OptimizationProblemModel1SingleHour:
    """
    Single hour myopic deterministic investment model:

    max  Σ_k (λ - c_k) α_k x_k + C_capex_fixed
    s.t. Σ_k x_k <= X_max
         x_k >= 0  for all k
    """
    def __init__(self, data: InputDataModel1SingleHour):
        self.data = data
        self.m = gp.Model("Model1_SingleHour")
        self.vars = Expando()
        self.results = Expando()
        self.con = Expando()

    def build(self):
        d = self.data
        K = d.tech

        # Decision variables
        x = self.m.addVars(K, lb=0.0, name="x")

        # Lower bounds
        self.con.x_lb = {}
        for k in K:
            self.con.x_lb[k] = self.m.addConstr(x[k] >= 0.0, name=f"x_lb[{k}]")

        # Max per technology: 200 MW for each
        max_cap = 200
        self.con.x_ub = {}
        for k in K:
            self.con.x_ub[k] = self.m.addConstr(x[k] <= max_cap, name=f"x_ub[{k}]")

        # Nuclear minimum: 100 MW
        #nuclear_index = d.tech_names.index("nuclear")
        #self.con.nuclear_min = self.m.addConstr(
            #x[nuclear_index] >= 100, name="nuclear_min"
        #)

        # Total capacity cap: 500 MW
        self.con.total_cap = self.m.addConstr(
            gp.quicksum(x[k] for k in K) <= d.X_max,
            name="total_capacity_cap"
        )

        # Objective
        obj_expr = gp.quicksum(
            (d.lambda_price - d.c[k]) * d.alpha[k] * x[k]
            for k in K
        ) + d.C_capex_fixed

        self.m.setObjective(obj_expr, GRB.MAXIMIZE)

        self.vars.x = x


    def solve(self, verbose: bool = False):
        if not verbose:
            self.m.Params.OutputFlag = 0
        self.m.optimize()

        if self.m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi status: {self.m.Status}")

        d = self.data
        x = self.vars.x

        self.results = Expando()
        self.results.x = np.array([x[k].X for k in d.tech])
        self.results.obj = self.m.ObjVal

        return self.results
    
if __name__ == "__main__": 
    import matplotlib.pyplot as plt

    lambda_price = 70.0  # €/MWh
    c_base = np.array([10, 45, 100, 60, 12])       # five techs
    alpha = np.array([0.50, 1, 1.0, 1.0, 1.0])     # e.g. wind, coal, oil, biomass, nuclear
    X_max = 500.0  # MW
    Capex_one_hour = -1000000000 / (25 * 365.25 * 24)  # 1 billion EUR capex 

    tech_names = ["wind", "coal", "oil", "biomass", "nuclear"]

    # Define scenarios: scale factors on marginal costs
    scenario_scales = [0.5, 1.0, 2.0]
    scenario_labels = ["Half c", "Base c", "Double c"]

    # Store optimal capacities for each scenario
    x_solutions = []

    for scale in scenario_scales:
        c = c_base * scale

        data = InputDataModel1SingleHour(
            lambda_price=lambda_price,
            c=c,
            alpha=alpha,
            X_max=X_max,
            C_capex_fixed=Capex_one_hour,
            tech_names=tech_names
        )

        prob = OptimizationProblemModel1SingleHour(data)
        prob.build()
        res = prob.solve()

        x_solutions.append(res.x)

        print(f"\nScenario with c scaled by {scale}:")
        for name, x_val in zip(tech_names, res.x):
            print(f"  Optimal {name:8s} capacity (MW): {x_val:6.1f}")
        print(f"  Objective (hourly profit): {res.obj:.2f}\n")

    # Convert to array of shape (n_tech, n_scenarios)
    x_solutions = np.vstack(x_solutions).T  # shape (5, 3) here

    # Plot grouped bar chart
    n_tech = len(tech_names)
    n_scen = len(scenario_scales)
    indices = np.arange(n_tech)
    width = 0.8 / n_scen  # total group width 0.8

    fig, ax = plt.subplots()

    for j in range(n_scen):
        ax.bar(
            indices + (j - (n_scen - 1) / 2) * width,
            x_solutions[:, j],
            width,
            label=scenario_labels[j],
        )

    ax.set_xticks(indices)
    ax.set_xticklabels(tech_names)
    ax.set_ylabel("Optimal capacity [MW]")
    ax.set_title("Optimal investment by marginal cost scenario")
    ax.legend()
    plt.grid('y')
    plt.tight_layout()
    plt.show()


'''
if __name__ == "__main__": 
    lambda_price = 70.0  # €/MWh
    c = np.array([10, 45, 100, 60, 12])       # five techs
    alpha = np.array([0.50, 1, 1.0, 1.0, 1.0])  # e.g. wind, coal, oil, biomass, nuclear
    X_max = 500.0  # MW
    Capex_one_hour = -1000000000 / (25*365.25*24)# 1 billion EUR capex 

    tech_names = ["wind", "coal", "oil", "biomass", "nuclear"]

    data = InputDataModel1SingleHour(
        lambda_price=lambda_price,
        c=c,
        alpha=alpha,
        X_max=X_max,
        C_capex_fixed=Capex_one_hour,
        tech_names=tech_names
    )

    prob = OptimizationProblemModel1SingleHour(data)
    prob.build()
    res = prob.solve()

    print("\nModel 1 Results (myopic model at standard price):")
    print("\nWind alpha coefficient:", alpha[0])
    print("\nOptimal Wind capacity     (MW): ", res.x[0])
    print("Optimal Coal capacity     (MW): ", res.x[1])
    print("Optimal Oil capacity      (MW): ", res.x[2])
    print("Optimal Biomass capacity  (MW): ", res.x[3])
    print("Optimal Nuclear capacity  (MW): ", res.x[4])
    print("\nObjective (hourly profit):", res.obj, "\n") '''
