
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euler(f, t0, y0, h, n):
    import numpy as np
    t = np.zeros(n+1)
    y = np.zeros((n+1, len(np.atleast_1d(y0))))
    t[0] = t0
    y[0] = np.atleast_1d(y0)
    for k in range(n):
        y[k+1] = y[k] + h * f(t[k], y[k])
        t[k+1] = t[k] + h
    return t, y.squeeze()

def rk4(f, t0, y0, h, n):
    import numpy as np
    t = np.zeros(n+1)
    y = np.zeros((n+1, len(np.atleast_1d(y0))))
    t[0] = t0
    y[0] = np.atleast_1d(y0)
    for k in range(n):
        tk, yk = t[k], y[k]
        k1 = f(tk, yk)
        k2 = f(tk + 0.5*h, yk + 0.5*h*k1)
        k3 = f(tk + 0.5*h, yk + 0.5*h*k2)
        k4 = f(tk + h, yk + h*k3)
        y[k+1] = yk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t[k+1] = tk + h
    return t, y.squeeze()

# Problem A: Logistic growth
r = 0.8
K = 10.0
y0_log = 1.0
t0, T = 0.0, 10.0
def f_logistic(t, y):
    import numpy as np
    y = np.atleast_1d(y)
    return r*y*(1 - y/K)
def y_logistic_exact(t):
    import numpy as np
    return K / (1 + ((K - y0_log)/y0_log)*np.exp(-r*t))

# Problem B: Simple Harmonic Oscillator
w = 2.0
A = 1.0
v0 = 0.0
y0_sho = np.array([A, v0])
t0_sho, T_sho = 0.0, 10.0
def f_sho(t, y):
    import numpy as np
    y = np.atleast_1d(y)
    return np.array([y[1], -w**2 * y[0]])
def y_sho_exact(t):
    import numpy as np
    y = A*np.cos(w*t) + (v0/w)*np.sin(w*t)
    yp = -A*w*np.sin(w*t) + v0*np.cos(w*t)
    return y, yp

def main():
    os.makedirs("project2_numerical", exist_ok=True)
    base_dir = "project2_numerical"

    # Time series for Logistic
    n_plot = 160
    h_plot = (T - t0)/n_plot
    t_eu, y_eu = euler(f_logistic, t0, y0_log, h_plot, n_plot)
    t_rk, y_rk = rk4(f_logistic, t0, y0_log, h_plot, n_plot)
    y_ex = y_logistic_exact(t_eu)

    plt.figure()
    plt.plot(t_eu, y_ex, label="Exact")
    plt.plot(t_eu, y_eu, label="Euler")
    plt.plot(t_rk, y_rk, label="RK4")
    plt.xlabel("t"); plt.ylabel("y(t)")
    plt.title("Logistic Growth: Exact vs Euler vs RK4")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_logistic_timeseries.png"), dpi=200)
    plt.close()

    # Convergence study for Logistic
    steps = [20, 40, 80, 160, 320]
    error_rows = []
    for n in steps:
        h = (T - t0)/n
        _, y_e = euler(f_logistic, t0, y0_log, h, n)
        _, y_r = rk4(f_logistic, t0, y0_log, h, n)
        err_e = abs(y_e[-1] - y_logistic_exact(T))
        err_r = abs(y_r[-1] - y_logistic_exact(T))
        error_rows.append({"method":"Euler","n_steps":n,"h":h,"error_at_T":err_e})
        error_rows.append({"method":"RK4","n_steps":n,"h":h,"error_at_T":err_r})
    import pandas as pd
    df_err = pd.DataFrame(error_rows)
    df_err.to_csv(os.path.join(base_dir, "logistic_errors.csv"), index=False)

    plt.figure()
    for method in ["Euler","RK4"]:
        sub = df_err[df_err["method"] == method].sort_values("h")
        plt.loglog(sub["h"], sub["error_at_T"], marker="o", label=method)
    plt.xlabel("Step size h (log scale)")
    plt.ylabel("Global error at T (log scale)")
    plt.title("Logistic Growth: Convergence of Euler vs RK4")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_logistic_error_convergence.png"), dpi=200)
    plt.close()

    # SHO time series and phase portrait
    n_plot_sho = 400
    h_plot_sho = (T_sho - t0_sho)/n_plot_sho
    t_eu_sho, y_eu_sho = euler(f_sho, t0_sho, y0_sho, h_plot_sho, n_plot_sho)
    t_rk_sho, y_rk_sho = rk4(f_sho, t0_sho, y0_sho, h_plot_sho, n_plot_sho)
    y_exact_sho, yp_exact_sho = y_sho_exact(t_eu_sho)

    plt.figure()
    plt.plot(t_eu_sho, y_exact_sho, label="Exact position")
    plt.plot(t_eu_sho, y_eu_sho[:,0], label="Euler position")
    plt.plot(t_rk_sho, y_rk_sho[:,0], label="RK4 position")
    plt.xlabel("t"); plt.ylabel("y(t)")
    plt.title("Simple Harmonic Oscillator: Position vs Time")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_sho_timeseries.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(y_exact_sho, yp_exact_sho, label="Exact")
    plt.plot(y_eu_sho[:,0], y_eu_sho[:,1], label="Euler")
    plt.plot(y_rk_sho[:,0], y_rk_sho[:,1], label="RK4")
    plt.xlabel("y"); plt.ylabel("y'")
    plt.title("Simple Harmonic Oscillator: Phase Portrait")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_sho_phase_portrait.png"), dpi=200)
    plt.close()

    # SHO convergence at T (position only)
    steps_sho = [40, 80, 160, 320, 640]
    error_rows_sho = []
    for n in steps_sho:
        h = (T_sho - t0_sho)/n
        _, y_e = euler(f_sho, t0_sho, y0_sho, h, n)
        _, y_r = rk4(f_sho, t0_sho, y0_sho, h, n)
        yT_exact, _ = y_sho_exact(T_sho)
        err_e = abs(y_e[-1,0] - yT_exact)
        err_r = abs(y_r[-1,0] - yT_exact)
        error_rows_sho.append({"method":"Euler","n_steps":n,"h":h,"error_at_T":err_e})
        error_rows_sho.append({"method":"RK4","n_steps":n,"h":h,"error_at_T":err_r})
    df_sho = pd.DataFrame(error_rows_sho)
    df_sho.to_csv(os.path.join(base_dir, "sho_errors.csv"), index=False)

    plt.figure()
    for method in ["Euler","RK4"]:
        sub = df_sho[df_sho["method"] == method].sort_values("h")
        plt.loglog(sub["h"], sub["error_at_T"], marker="o", label=method)
    plt.xlabel("Step size h (log scale)"); plt.ylabel("Global error at T (log scale)")
    plt.title("SHO: Convergence of Euler vs RK4")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_sho_error_convergence.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
