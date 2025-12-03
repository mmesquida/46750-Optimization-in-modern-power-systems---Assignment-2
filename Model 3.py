# Model_3.py

import numpy as np
import gurobipy as gp
from gurobipy import GRB


gp.setParam("LogToConsole", 0)


class Expando:
    pass


def simulate_capex_auction(base_capex, base_capex_per_mw, n_units=10, n_buyers=10, seed=42):
    np.random.seed(seed)
    base_capex_per_mw = np.asarray(base_capex_per_mw, dtype=float)
    K = len(base_capex_per_mw)
    C_capex = np.zeros(K)

    print("\nBase CAPEX per MW for auction simulation:", base_capex)
    print("Number of generators up for auction:", n_units)
    print("Number of buyers in auction:", n_buyers)

    for k in range(K):
        # Seller asks (ascending)
        asks = base_capex_per_mw[k] * np.random.uniform(0.8, 1.2, size=n_units)
        asks = np.sort(asks)

        # Buyer bids (ascending)
        bids = base_capex_per_mw[k] * np.random.lognormal(mean=0.4, sigma=0.5, size=n_buyers)
        bids = np.sort(bids)

        # Number of trades
        trades = min(n_units, n_buyers)

        # Clearing price = highest losing bid (scarcity drives this up)
        if n_buyers > trades:
            clearing_price = bids[-(trades + 1)]  # highest losing bid
        else:
            clearing_price = max(asks[trades - 1], bids[-1])

        C_capex[k] = clearing_price

    print("\nCAPEX Auction Results:")
    print("Wind - Asking price", base_capex_per_mw[0], "€/MW", 
          "-> Clearing price:", C_capex[0], "€/MW -> ratio:", C_capex[0]/base_capex_per_mw[0])
    print("Coal - Asking price", base_capex_per_mw[1], "€/MW", 
          "-> Clearing price:", C_capex[1], "€/MW -> ratio:", C_capex[1]/base_capex_per_mw[1])
    print("Oil - Asking price", base_capex_per_mw[2], "€/MW", 
          "-> Clearing price:", C_capex[2], "€/MW -> ratio:", C_capex[2]/base_capex_per_mw[2])
    print("Biomass - Asking price", base_capex_per_mw[3], "€/MW", 
          "-> Clearing price:", C_capex[3], "€/MW -> ratio:", C_capex[3]/base_capex_per_mw[3])
    print("Nuclear - Asking price", base_capex_per_mw[4], "€/MW", 
          "-> Clearing price:", C_capex[4], "€/MW -> ratio:", C_capex[4]/base_capex_per_mw[4])
    

    return C_capex



class InputDataModel3:
    def __init__(
        self,
        lambdas,
        c,
        alpha,
        C_capex,
        X_max,
        x_ub,
        tech_names,
        x_min_nuc=0.0,
    ):
        lambdas = np.asarray(lambdas, dtype=float)
        c = np.asarray(c, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        C_capex = np.asarray(C_capex, dtype=float)
        x_ub = np.asarray(x_ub, dtype=float)

        assert lambdas.ndim == 1
        assert c.ndim == 2
        assert c.shape[1] == lambdas.shape[0]
        assert c.shape[0] == len(alpha) == len(C_capex) == len(x_ub) == len(tech_names)

        self.lambdas = lambdas
        self.c = c
        self.alpha = alpha
        self.C_capex = C_capex

        self.n_tech = c.shape[0]
        self.T = lambdas.shape[0]
        self.tech = list(range(self.n_tech))
        self.tech_names = list(tech_names)

        self.X_max = float(X_max)
        self.x_ub = x_ub
        self.x_min_nuc = float(x_min_nuc)


class OptimizationProblemModel3:
    def __init__(self, data: InputDataModel3):
        self.data = data
        self.m = gp.Model("Model3_24h")
        self.vars = Expando()
        self.con = Expando()
        self.results = Expando()

    def build(self):
        d = self.data
        K = d.tech

        x = self.m.addVars(K, lb=0.0, name="x")

        self.con.x_ub = {}
        for k in K:
            self.con.x_ub[k] = self.m.addConstr(x[k] <= d.x_ub[k], name=f"x_ub[{k}]")

        lower_names = [name.lower() for name in d.tech_names]
        if d.x_min_nuc > 0.0 and "nuclear" in lower_names:
            nuc_idx = lower_names.index("nuclear")
            self.con.x_nuc_min = self.m.addConstr(
                x[nuc_idx] >= d.x_min_nuc, name="x_nuclear_min"
            )

        self.con.total_cap = self.m.addConstr(
            gp.quicksum(x[k] for k in K) <= d.X_max, name="total_capacity_cap"
        )

        m_k = np.zeros(d.n_tech)
        for k in K:
            margin_24h = np.sum(d.lambdas - d.c[k, :]) * d.alpha[k]
            m_k[k] = margin_24h

        obj = gp.quicksum((m_k[k] - d.C_capex[k]) * x[k] for k in K)
        self.m.setObjective(obj, GRB.MAXIMIZE)

        self.vars.x = x
        self.results.m_k = m_k

    def solve(self, verbose: bool = True):
        self.m.optimize()

        if self.m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi status: {self.m.Status}")

        d = self.data
        x = self.vars.x

        x_sol = np.array([x[k].X for k in d.tech])
        self.results.x = x_sol
        self.results.obj = self.m.ObjVal
        onehour_equiv_obj = self.results.obj / 24

        if verbose:
            print(f"\nOptimal objective value (24h): {self.results.obj:.2f}")
            print(f"Optimal objective value (1h equivalent): {onehour_equiv_obj:.2f}")
            print("Optimal capacities x_k [MW]:")
            for name, val in zip(d.tech_names, x_sol):
                print(f"  {name}: {val:.2f} MW")

        return self.results


if __name__ == "__main__":
    lambda_price = 70.0   # €/MWh, price taker
    T = 24
    lambdas = np.full(T, lambda_price)

    tech_names = ["wind", "coal", "oil", "biomass", "nuclear"]
    n_tech = len(tech_names)

    c_base = np.array([10.0, 45.0, 100.0, 60.0, 12.0])   # €/MWh
    alpha = np.array([0.50, 1.0, 1.0, 1.0, 1.0])

    c = np.tile(c_base.reshape(n_tech, 1), (1, T))

    base_capex = 260 # €/MW
    base_capex_per_mw = np.array([0.6 * base_capex, # wind cheaper relative to its margin
                                  0.4 * base_capex, # coal cheap
                                  1.0 * base_capex, # oil insanely expensive -> never used
                                  0.5 * base_capex, # biomass cheap
                                  1 * base_capex])  # nuclear expensive relative to its margin


    C_capex = simulate_capex_auction(
        base_capex_per_mw=base_capex_per_mw,
        n_units=5,
        n_buyers=750,
        seed=42,
        base_capex=base_capex,
    )

    X_max = 500.0
    x_ub = np.full(n_tech, 200.0)
    x_min_nuc = 0.0  # same nuclear min as Model 1

    data = InputDataModel3(
        lambdas=lambdas,
        c=c,
        alpha=alpha,
        C_capex=C_capex,
        X_max=X_max,
        x_ub=x_ub,
        tech_names=tech_names,
        x_min_nuc=0,
    )

    print("\nWind alpha coefficient:", alpha[0])

    prob = OptimizationProblemModel3(data)
    prob.build()
    prob.solve(verbose=True)
