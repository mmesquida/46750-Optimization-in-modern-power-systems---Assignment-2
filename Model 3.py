# Model_3.py

import numpy as np
import gurobipy as gp
from gurobipy import GRB


gp.setParam("LogToConsole", 0)


class Expando:
    pass


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

    # No competition: return asking price exactly
    if n_buyers <= 1:
        return base_capex_per_mw.copy()

    # Competition factor
    delta = n_buyers - 1
    factor = 1 + competition_intensity * (1 - np.exp(-delta / saturation_scale))

    # Small iid noise around the scaling
    noise = 1 + noise_level * (np.random.uniform(-1, 1, size=len(base_capex_per_mw)))

    return base_capex_per_mw * factor * noise





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

    import matplotlib.pyplot as plt

    # --- Market environment ---
    lambda_price = 70.0   # €/MWh, price taker
    T = 24
    lambdas = np.full(T, lambda_price)

    tech_names = ["wind", "coal", "oil", "biomass", "nuclear"]
    n_tech = len(tech_names)

    # Marginal costs and availability factors
    c_base = np.array([10.0, 45.0, 100.0, 60.0, 12.0])   # €/MWh
    alpha  = np.array([0.50, 1.0, 1.0, 1.0, 1.0])
    c = np.tile(c_base.reshape(n_tech, 1), (1, T))

    # --- Baseline CAPEX (asking price) ---
    base_capex = 300  # €/MW
    base_capex_per_mw = np.array([
        0.6 * base_capex,  # wind
        0.4 * base_capex,  # coal
        0.8 * base_capex,  # oil
        0.5 * base_capex,  # biomass
        1.0 * base_capex   # nuclear
    ])

    # === CAPEX vs number of competitors (for tuning / plotting) ===
    max_buyers = 20          # plot from 1 to max_buyers
    all_C = np.zeros((max_buyers, n_tech))
    seed_base = 42           # base seed for noise

    for n in range(1, max_buyers + 1):
        C_capex_n = simulate_capex_auction(
            base_capex_per_mw=base_capex_per_mw,
            n_buyers=n,
            competition_intensity=4,   # tune max uplift
            saturation_scale=5,        # tune how fast it rises
            noise_level=0.05,          # small random deviation
            seed=seed_base + n         # reproducible per n, each n gets different noise
        )
        all_C[n-1, :] = C_capex_n

    # Plot CAPEX curves for each technology
    phi = (1+np.sqrt(5))/2  # Calculate golden ratio for correct proposrtions
    x_length = 10
    y_length = x_length / phi
    plt.figure(figsize=(x_length, y_length))
    buyers_range = np.arange(1, max_buyers + 1)
    for k, tech in enumerate(tech_names):
        plt.plot(buyers_range, all_C[:, k], label=tech)

    plt.xlabel("Number of competitors (n buyers)")
    plt.ylabel("CAPEX per MW [€/MW]")
    plt.title("Auction-derived CAPEX as competition increases")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(np.arange(1, max_buyers + 1, 1))
    plt.tight_layout()
    plt.show()

    # === NEW: net margin vs competitors (break-even plot, FIGURE 2) ===
    # 24h margins m_k per MW (same formula as in OptimizationProblemModel3.build)
    m_k = np.zeros(n_tech)
    for k in range(n_tech):
        margin_24h = np.sum(lambdas - c[k, :]) * alpha[k]
        m_k[k] = margin_24h

    net_margin = m_k[None, :] - all_C  # shape (max_buyers, n_tech)

    plt.figure(figsize=(x_length, y_length))
    for k, tech in enumerate(tech_names):
        plt.plot(buyers_range, net_margin[:, k], label=tech)

        # mark first n where net margin becomes <= 0 (break-even)
        crossing = np.where(net_margin[:, k] <= 0)[0]
        if crossing.size > 0:
            n_star = buyers_range[crossing[0]]
            plt.scatter(n_star, net_margin[crossing[0], k], s=40)

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Number of competitors (n buyers)")
    plt.ylabel("Net 24h margin per MW [€/MW]")
    plt.title("Break-even competition level for each technology")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(np.arange(1, max_buyers + 1, 1))
    plt.tight_layout()
    plt.show()

    # --- Choose number of competing buyers for the optimisation run ---
    n_buyers_run = 10  # pick one scenario for Model 3

    C_capex = simulate_capex_auction(
        base_capex_per_mw=base_capex_per_mw,
        n_buyers=n_buyers_run,
        competition_intensity=4,
        saturation_scale=5,
        noise_level=0.05,
        seed=42
    )

    # --- Capacity limits ---
    X_max = 500.0
    x_ub  = np.full(n_tech, 200.0)  # per-technology upper bound
    x_min_nuc = 0.0   # no nuclear minimum

    # --- Construct input data ---
    data = InputDataModel3(
        lambdas=lambdas,
        c=c,
        alpha=alpha,
        C_capex=C_capex,
        X_max=X_max,
        x_ub=x_ub,
        tech_names=tech_names,
        x_min_nuc=x_min_nuc,
    )

    print("\nWind alpha coefficient:", alpha[0])
    print("CAPEX used in optimisation:", C_capex)
    print("Base CAPEX:", base_capex)
    print("Wind asking price:", base_capex_per_mw[0], "-> Auction price:", C_capex[0])
    print("Coal asking price:", base_capex_per_mw[1], "-> Auction price:", C_capex[1])
    print("Oil asking price:", base_capex_per_mw[2], "-> Auction price:", C_capex[2])
    print("Biomass asking price:", base_capex_per_mw[3], "-> Auction price:", C_capex[3])
    print("Nuclear asking price:", base_capex_per_mw[4], "-> Auction price:", C_capex[4])

    # --- Solve Model 3 ---
    prob = OptimizationProblemModel3(data)
    prob.build()
    prob.solve(verbose=True)


