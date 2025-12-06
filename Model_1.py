"""
Model 1 - Myopic Deterministic Investment Model
-----------------------------------------------

This script implements the simplest version of the investment problem, where
profit is computed for a single representative hour with fixed price, fixed
marginal costs, and a fixed CAPEX. The model determines how much capacity to
invest in for each technology under idealised, certainty-equivalent conditions.

What the script does:
• Defines λ (price), c_i (marginal costs), α_i (availability), and CAPEX.
• Formulates and solves the linear optimisation:
        maximize  Σ_i α_i (λ - c_i) x_i - CAPEX
        subject to Σ_i x_i ≤ X_max  and  0 ≤ x_i ≤ x_{i,max}
• Prints the optimal investment vector x_i.

Key parameters to adjust:
• λ         electricity price
• c_i       marginal costs per technology
• α_i       availability (e.g., wind derating)
• X_max     total investment limit
• x_ub      per-technology upper bounds

USAGE:
Run the script directly to obtain the optimal capacity mix under the model’s
simple, deterministic assumptions.
"""

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
    CF        : 1D array length n_tech with availability factors α_k (0..1)
                   e.g. α_wind, α_solar < 1; others = 1
    X_max        : total capacity budget (MW)
    C_capex_fixed: fixed capex term (can be zero if only argmax matters)
    """
    def __init__(
        self,
        lambda_price: float,
        c: np.ndarray,
        CF: np.ndarray,
        X_max: float,
        C_capex_fixed: float = 0.0,
        tech_names: Optional[List[str]] = None,
    ):
        assert c.ndim == 1, "c must be 1D (length n_tech)"
        assert CF.ndim == 1, "CF must be 1D (length n_tech)"
        assert len(c) == len(CF), "c and CF must have same length"

        self.lambda_price = float(lambda_price)
        self.c = c.astype(float)
        self.CF = CF.astype(float)
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

        # Total capacity cap: 500 MW
        self.con.total_cap = self.m.addConstr(
            gp.quicksum(x[k] for k in K) <= d.X_max,
            name="total_capacity_cap"
        )

        # Objective
        obj_expr = gp.quicksum(
            (d.lambda_price - d.c[k]) * d.CF[k] * x[k]
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
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Arial"     # or "Arial", "Times New Roman", "Calibri"

    lambda_price = 70.0  # €/MWh
    c_base = np.array([10, 45, 100, 60, 12])       # five techs
    CF = np.array([0.50, 1, 1.0, 1.0, 1.0])     # e.g. wind, coal, oil, biomass, nuclear
    X_max = 500.0  # MW
    Capex_one_hour = -1000000000 / (25 * 365.25 * 24)  # 1 billion EUR capex 

    tech_names = ["wind", "coal", "oil", "biomass", "nuclear"]

    # Define scenarios: scale factors on marginal costs
    #scenario_scales = [0.5, 1.0, 2.0]
    #scenario_labels = ["Half c", "Base c", "Double c"]

    # Scenario definitions
    scenario_labels = ["Base", "Wind Double", "Biomass half", "Oil half"]

    # Build c arrays for each scenario
    c_scenarios = []

    # 1. Base case
    c_scenarios.append(c_base.copy())

    # 2. Wind Double
    c2 = c_base.copy()
    c2[0] = 2 * c2[0] # wind index = 0
    c_scenarios.append(c2)

    # 3. Biomass half cost
    c3 = c_base.copy()
    c3[3] = 0.5 * c3[3]     # biomass index = 3
    c_scenarios.append(c3)

    # 4. Oil half cost
    c4 = c_base.copy()
    c4[2] = 0.5 * c4[2]     # oil index = 2
    c_scenarios.append(c4)

    print(c_scenarios)

    # Store optimal capacities for each scenario
    x_solutions = []

    x_solutions = []

    i = 1
    for c_vec in c_scenarios:
        data = InputDataModel1SingleHour(
            lambda_price=lambda_price,
            c=c_vec,
            CF=CF,
            X_max=X_max,
            C_capex_fixed=Capex_one_hour,
            tech_names=tech_names
        )

        prob = OptimizationProblemModel1SingleHour(data)
        prob.build()
        res = prob.solve()

        x_solutions.append(res.x)

        if i == 1:  
            print(f"\n Base Scenario: ")
        else:   
            print(f"\n Scenario {i}:")
        i += 1
        for name, x_val in zip(tech_names, res.x):
            print(f"  Optimal {name:8s} capacity (MW): {x_val:6.1f}")
        print(f"  Objective (hourly profit): {res.obj:.2f}\n")

    # Convert to array of shape (n_tech, n_scenarios)
    x_solutions = np.vstack(x_solutions).T  # shape (5, 3) here

    import matplotlib.pyplot as plt
    import numpy as np

    # Original order from your model
    tech_names = ["Wind", "Coal", "Oil", "Biomass", "Nuclear"]
    scenario_labels = ["Base Case", "Scenario 2 $(2 c_{wind})$", "Scenario 3 $(0.5 c_{biomass})$", "Scenario 4 $(0.5 c_{oil})$"]

    # x_solutions shape = (n_tech, n_scen) in original tech order
    # Example: x_solutions[k, s], where k matches tech_names[k]

    # Choose your stacking order
    desired_order = ["Nuclear", "Wind", "Coal", "Biomass", "Oil"]

    # Determine indices in x_solutions that match this order
    ordered_indices = [tech_names.index(t) for t in desired_order]

    # Reorder arrays
    x_plot = x_solutions[ordered_indices, :]
    tech_plot_names = [tech_names[i] for i in ordered_indices]

    # Golden ratio aspect
    golden_ratio = (1 + 5**0.5) / 2
    width = 10
    height = width / golden_ratio

    colors = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#E15759",  # red
    "#76B7B2",  # teal
    "#59A14F",  # green
]






    fig, ax = plt.subplots(figsize=(width, height))

    indices = np.arange(x_plot.shape[1])   # 3 scenarios

    bottom = np.zeros(x_plot.shape[1])

    # Choose bar width (0.4 looks clean for stacked bars)
    bar_width = 0.6

    for k, tech in enumerate(tech_plot_names):
        ax.bar(
            indices,
            x_plot[k, :],
            bottom=bottom,
            width=bar_width,
            label=tech,
            color=colors[k],
        )
        bottom += x_plot[k, :]

    # Nice axis labels
    ax.set_xticks(indices)
    ax.set_xticklabels(scenario_labels, fontsize = 17)
    ax.set_ylabel("Optimal Capacity [MW]", fontsize = 17)
    ax.set_title("Optimal Technology mix by Scenario", fontsize = 23)
    ax.tick_params(axis="both", labelsize=17)


    # Add horizontal grid lines *behind* the bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylim(0,550)

    # Legend
    ax.legend( loc="upper right", fontsize = 15, ncol = 5)

    plt.tight_layout()
    plt.show()

