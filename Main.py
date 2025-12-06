
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from Model_2 import solve_model2


if __name__ == "__main__":

    run_model = 1 # Change this value to 1, 2, or 3 to run different models

    if run_model == 1:

        T = 24
        K = 5
        hours = np.arange(T)

        # 1) Demand D profile
        base_load = 500
        morning_peak = 150 * np.exp(-0.5 * ((hours - 8) / 3)**2)
        evening_peak = 250 * np.exp(-0.5 * ((hours - 19) / 3)**2)
        D = base_load + morning_peak + evening_peak
        D = D.round(1)

        # 2) Prices λ_i 
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