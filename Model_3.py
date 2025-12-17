"""
Model 3 — Investment Under Competition-Driven CAPEX Formation

This script implements Model 3 from the assignment: a 24-hour investment 
optimisation model where the CAPEX of each technology depends on the level 
of competition in the generator acquisition market. A reduced-form auction 
model generates technology-specific CAPEX curves as the number of competing 
bidders increases. The script:

1. Builds a realistic 24-hour market environment (demand, prices, wind profile)
2. Simulates competition-adjusted CAPEX for 1…N bidders
3. Plots CAPEX growth and break-even points for each technology
4. Solves the optimisation for a chosen competition level using Gurobi

Usage:
------
All outputs (plots + optimisation results) are produced automatically. No 
arguments are required.

Key parameters to experiment with:
----------------------------------
- base_capex_per_mw : baseline cost of each technology
- competition_intensity (gamma) : how strongly CAPEX increases with competition
- saturation_scale (s)      : how quickly the CAPEX curve flattens
- noise_level               : random variation in CAPEX bids
- max_buyers               : range of bidders for CAPEX/break-even plots
- n_buyers_run             : number of bidders used for the final optimisation

These allow exploration of how competitive pressure affects CAPEX formation,
technology attractiveness, and the resulting optimal investment mix.
"""

from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter    
import numpy as np
import gurobipy as gp
from gurobipy import GRB

import matplotlib as mpl

mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\Rasmu\anaconda3\envs\integrated-energy-grids\Library\bin\ffmpeg.exe"


gp.setParam("LogToConsole", 0)

# Simple class to hold attributes
class Expando:
    pass

# Simulate CAPEX per MW as a function of number of buyers in auction
def simulate_capex_auction(
    base_capex_per_mw,
    n_buyers,
    competition_intensity=0.10,
    saturation_scale=10.0, # how fast the competition effect saturates
    noise_level=0.05, # relative noise level
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




# Input data class for Model 3
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

        assert alpha.shape == c.shape, "alpha must be of shape (n_tech, T)"

        n_tech = c.shape[0]
        assert len(C_capex) == n_tech
        assert len(x_ub) == n_tech
        assert len(tech_names) == n_tech

        self.lambdas = lambdas
        self.c = c
        self.alpha = alpha
        self.C_capex = C_capex

        self.n_tech = n_tech
        self.T = lambdas.shape[0]
        self.tech = list(range(self.n_tech))
        self.tech_names = list(tech_names)

        self.X_max = float(X_max)
        self.x_ub = x_ub
        self.x_min_nuc = float(x_min_nuc)


# Optimization problem class for Model 3
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
        # Per-technology capacity upper bounds
        for k in K:
            self.con.x_ub[k] = self.m.addConstr(x[k] <= d.x_ub[k], name=f"x_ub[{k}]")

        # Minimum nuclear capacity constraint
        lower_names = [name.lower() for name in d.tech_names]
        if d.x_min_nuc > 0.0 and "nuclear" in lower_names:
            nuc_idx = lower_names.index("nuclear")
            self.con.x_nuc_min = self.m.addConstr(
                x[nuc_idx] >= d.x_min_nuc, name="x_nuclear_min"
            )

        # Total capacity upper bound
        self.con.total_cap = self.m.addConstr(
            gp.quicksum(x[k] for k in K) <= d.X_max, name="total_capacity_cap"
        )
        # Objective: max sum_k (m_k - C_capex_k) * x_k
        m_k = np.zeros(d.n_tech)
        for k in K:
            margin_24h = np.sum((d.lambdas - d.c[k, :]) * d.alpha[k, :])
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
    T = 24
    hours = np.arange(T)

    # Demand curve as in Model 2
    base_load = 300
    morning_peak = 150 * np.exp(-0.5 * ((hours - 8) / 3)**2)
    evening_peak = 250 * np.exp(-0.5 * ((hours - 19) / 3)**2)
    D = base_load + morning_peak + evening_peak
    factor = 500 / D.max()
    D = (D * factor).round(1)

    # Price curve λ_t as in Model 2
    rng_prices = np.random.default_rng(2025)  #independent reproducible seed
    lambdas = 60 + 0.09 * (D - D.min())
    lambdas += rng_prices.normal(0, 3, T)
    lambdas = np.clip(lambdas, 20, 130)
    lambdas = lambdas.round(2)


    tech_names = ["wind", "coal", "oil", "biomass", "nuclear"]
    n_tech = len(tech_names)

    # Marginal costs and availability factors
    c_base = np.array([10.0, 45.0, 100.0, 60.0, 12.0])   # €/MWh

    # Wind capacity factor as in updated Model 2
    rng_cf = np.random.default_rng(seed=123)
    CF_wind = 0.7 + 0.37 * np.sin(2 * np.pi * (hours - 3) / 24)
    CF_wind += rng_cf.normal(0.0, 0.10, size=hours.shape[0])
    CF_wind = np.clip(CF_wind, 0.0, 1.0)

    alpha_time = np.ones((n_tech, T))
    idx_wind = tech_names.index("wind")
    alpha_time[idx_wind, :] = CF_wind


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

    # --- CAPEX vs number of competitors (for tuning / plotting) ---
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


    # === Animation for CAPEX vs competition curve ===
    fig_anim = plt.figure(figsize=(x_length, y_length))
    ax_anim = plt.gca()

    buyers_range = np.arange(1, max_buyers + 1)

    # Match your axis limits
    ax_anim.set_xlim(1, max_buyers)
    ax_anim.set_ylim(0, np.max(all_C) * 1.1)

    # Same labels & title
    ax_anim.set_xlabel("Number of competitors (n buyers)")
    ax_anim.set_ylabel("CAPEX per MW [€/MW]")
    ax_anim.set_title("Auction-derived CAPEX as competition increases")

    # Match your grid style
    ax_anim.grid(True, alpha=0.3)

    # Same integer ticks as static plot
    ax_anim.set_xticks(np.arange(1, max_buyers + 1, 1))

    # Prepare animated lines
    lines = []
    for k, tech in enumerate(tech_names):
        (ln,) = ax_anim.plot([], [], label=tech)
        lines.append(ln)

    ax_anim.legend(loc="upper left")

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def update(frame):
        # frame = number of buyers to show
        x = buyers_range[:frame]
        for k, ln in enumerate(lines):
            y = all_C[:frame, k]
            ln.set_data(x, y)
        return lines

    anim = FuncAnimation(
        fig_anim,
        update,
        frames=max_buyers + 1,   # +1 ensures frame 0 = blank
        init_func=init,
        interval=200,             # ms between frames
        blit=True
    )

    writer = FFMpegWriter(fps=3, bitrate=2400)
    anim.save("Competition Animation.mp4", writer=writer)
    print("Saved MP4 animation.")

    plt.close(fig_anim)


    # --- Break-even analysis ---
    # 24h margins m_k per MW (same formula as in OptimizationProblemModel3.build)
    m_k = np.zeros(n_tech, dtype=float)
    for k in range(n_tech):
        margin_24h = np.sum((lambdas - c[k, :]) * alpha_time[k, :])
        m_k[k] = float(margin_24h)

    net_margin = m_k[None, :] - all_C  # shape (max_buyers, n_tech)

    plt.figure(figsize=(x_length, y_length))
    buyers_range = np.arange(1, max_buyers + 1)
    # Plot net margin curves and break-even points
    for k, tech in enumerate(tech_names):
        y = net_margin[:, k]
        plt.plot(buyers_range, y, label=tech)

        # --- break-even point via sign change & interpolation ---
        signs = np.sign(y)

        # Case 1: exactly zero at some integer n
        idx_zero = np.where(y == 0.0)[0]

        if idx_zero.size > 0:
            i0 = idx_zero[0]
            n_star = buyers_range[i0]
            plt.scatter(n_star, 0.0, s=30, color="black", zorder=5)

        else:
            # Case 2: sign change between two consecutive points
            sign_change_idx = np.where(signs[:-1] * signs[1:] < 0)[0]

            if sign_change_idx.size > 0:
                i0 = sign_change_idx[0]        # first segment where sign changes
                x0, x1 = buyers_range[i0], buyers_range[i0 + 1]
                y0, y1 = y[i0], y[i0 + 1]

                # linear interpolation: solve y(x) = 0 between (x0,y0) and (x1,y1)
                n_star = x0 - y0 * (x1 - x0) / (y1 - y0)

                # plot the point exactly on the zero line
                plt.scatter(n_star, 0.0, s=30, color="black", zorder=5)
        # If no zero and no sign change: no break-even dot for this tech

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Number of competitors (n buyers)")
    plt.ylabel("Net 24h margin per MW [€/MW]")
    plt.title("Break-even competition level for each technology")
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(1, max_buyers + 1, 1))

    # --- custom legend entry for break-even marker ---
    handles, labels = plt.gca().get_legend_handles_labels()
    break_even_handle = Line2D(
        [], [],
        linestyle="--",   # dashed line in legend
        marker="o",
        color="black",
        label="Break-even point",
    )
    handles.append(break_even_handle)
    labels.append("Break-even point")

    plt.legend(handles, labels)
    plt.tight_layout()
    plt.show()

    # --- Animation for break-even analysis ---
    buyers_range = np.arange(1, max_buyers + 1)

    # Precompute break-even points (n_star) and when they become visible (frame index)
    break_even_info = []  # list of (n_star, frame_index_to_show) or (None, None)
    for k in range(n_tech):
        y = net_margin[:, k]
        signs = np.sign(y)

        idx_zero = np.where(y == 0.0)[0]
        if idx_zero.size > 0:
            i0 = idx_zero[0]
            n_star = buyers_range[i0]
            frame_show = i0 + 1  # when that point is included
            break_even_info.append((n_star, frame_show))
        else:
            sign_change_idx = np.where(signs[:-1] * signs[1:] < 0)[0]
            if sign_change_idx.size > 0:
                i0 = sign_change_idx[0]
                x0, x1 = buyers_range[i0], buyers_range[i0 + 1]
                y0, y1 = y[i0], y[i0 + 1]
                # linear interpolation for n* where y=0
                n_star = x0 - y0 * (x1 - x0) / (y1 - y0)
                frame_show = i0 + 2  # need both points visible
                break_even_info.append((n_star, frame_show))
            else:
                break_even_info.append((None, None))  # no break-even for this tech

    fig_be = plt.figure(figsize=(x_length, y_length))
    ax_be = plt.gca()

    ax_be.set_xlim(1, max_buyers)
    ax_be.set_ylim(np.min(net_margin) * 1.1, np.max(net_margin) * 1.1)
    ax_be.set_xlabel("Number of competitors (n buyers)")
    ax_be.set_ylabel("Net 24h margin per MW [€/MW]")
    ax_be.set_title("Break-even competition level for each technology")
    ax_be.grid(True, alpha=0.3)
    ax_be.set_xticks(np.arange(1, max_buyers + 1, 1))

    # Horizontal zero line
    ax_be.axhline(0.0, color="black", linestyle="--", linewidth=1)

    # Prepare lines and scatter markers
    lines = []
    break_even_scatters = []
    for k, tech in enumerate(tech_names):
        (ln,) = ax_be.plot([], [], label=tech)
        lines.append(ln)
        scat = ax_be.scatter([], [], s=30, color="black", zorder=5)
        break_even_scatters.append(scat)

    # Custom legend entry for break-even point
    handles, labels = ax_be.get_legend_handles_labels()
    break_even_handle = Line2D(
        [], [], linestyle="--", marker="o",
        color="black", label="Break-even point"
    )
    handles.append(break_even_handle)
    labels.append("Break-even point")
    ax_be.legend(handles, labels, loc="best")

    def init_be():
        for ln in lines:
            ln.set_data([], [])
        for scat in break_even_scatters:
            scat.set_offsets(np.empty((0, 2)))
        return lines + break_even_scatters

    def update_be(frame):
        # frame = number of buyers to show
        x = buyers_range[:frame]
        for k in range(n_tech):
            y = net_margin[:frame, k]
            lines[k].set_data(x, y)

            n_star, frame_show = break_even_info[k]
            if n_star is not None and frame >= frame_show:
                break_even_scatters[k].set_offsets(np.array([[n_star, 0.0]]))
            else:
                break_even_scatters[k].set_offsets(np.empty((0, 2)))
        return lines + break_even_scatters

    anim_be = FuncAnimation(
        fig_be,
        update_be,
        frames=max_buyers + 1,     # frame 0 = blank, then 1..max_buyers
        init_func=init_be,
        interval=200,
        blit=True,
    )

    # Save as MP4 (uses the ffmpeg path you already configured)
    writer_be = FFMpegWriter(fps=3, bitrate=2400)
    anim_be.save("BreakEven Animation.mp4", writer=writer_be)
    plt.close(fig_be)
    print("Saved animation: BreakEven Animation.mp4")


    # --- Select competition level for optimisation output---
    n_buyers_run = 19  # one scenario for the optimisation
    C_capex = simulate_capex_auction(
        base_capex_per_mw=base_capex_per_mw,
        n_buyers=n_buyers_run,
        competition_intensity=4,
        saturation_scale=5,
        noise_level=0.05,
        seed=seed_base + n_buyers_run
    )

    # --- Capacity limits ---
    X_max = 500.0
    x_ub  = np.full(n_tech, 200.0)  # per-technology upper bound
    x_min_nuc = 0.0   # no nuclear minimum

    # --- Construct input data ---
    data = InputDataModel3(
        lambdas=lambdas,
        c=c,
        alpha=alpha_time,
        C_capex=C_capex,
        X_max=X_max,
        x_ub=x_ub,
        tech_names=tech_names,
        x_min_nuc=x_min_nuc,
    )


    print("CAPEX used in optimisation:", C_capex)
    print("Base CAPEX:", base_capex)
    print("Wind asking price:", base_capex_per_mw[0], "-> Auction price:", C_capex[0], "ratio:", C_capex[0]/base_capex_per_mw[0])
    print("Coal asking price:", base_capex_per_mw[1], "-> Auction price:", C_capex[1], "ratio:", C_capex[1]/base_capex_per_mw[1])
    print("Oil asking price:", base_capex_per_mw[2], "-> Auction price:", C_capex[2], "ratio:", C_capex[2]/base_capex_per_mw[2])
    print("Biomass asking price:", base_capex_per_mw[3], "-> Auction price:", C_capex[3], "ratio:", C_capex[3]/base_capex_per_mw[3])
    print("Nuclear asking price:", base_capex_per_mw[4], "-> Auction price:", C_capex[4], "ratio:", C_capex[4]/base_capex_per_mw[4])

    # --- Solve Model 3 ---
    prob = OptimizationProblemModel3(data)
    prob.build()
    prob.solve(verbose=True)


