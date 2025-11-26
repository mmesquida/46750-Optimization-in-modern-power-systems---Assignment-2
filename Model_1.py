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

        # Decision variables: capacity for each technology (MW)
        x = self.m.addVars(K, lb=0.0, name="x")

        # Lower bound constraints (explicit, so you can read duals if needed)
        self.con.x_lb = {}
        for k in K:
            self.con.x_lb[k] = self.m.addConstr(x[k] >= 0.0, name=f"x_lb[{k}]")

        # Total capacity budget
        self.con.total_cap = self.m.addConstr(
            gp.quicksum(x[k] for k in K) <= d.X_max,
            name="total_capacity_cap"
        )

        # Objective: profit in this hour
        obj_expr = gp.quicksum(
            (d.lambda_price - d.c[k]) * d.alpha[k] * x[k]
            for k in K
        ) + d.C_capex_fixed

        self.m.setObjective(obj_expr, GRB.MAXIMIZE)

        # Save variable handles
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
    lambda_price = 60.0  # €/MWh
    c = np.array([20, 25, 40, 45, 55])       # five techs
    alpha = np.array([1, 1, 1.0, 1.0, 1.0])  # e.g. wind, coal, oil, waste, nuclear
    X_max = 500.0  # MWh

    data = InputDataModel1SingleHour(lambda_price, c, alpha, X_max)
    prob = OptimizationProblemModel1SingleHour(data)
    prob.build()
    res = prob.solve()

    print("Optimal capacities x:", res.x)
    print("Objective (hourly profit):", res.obj)

