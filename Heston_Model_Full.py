import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# ========== MARKET DATA ==========
S0 = 0.87           # Spot price
K = 0.95            # Strike
today = datetime(2025, 2, 6)
expiry = datetime(2025, 12, 17)
T = (expiry - today).days / 365.25  # Time to expiration in years
r = 0.04354         # Risk-free rate
q = 0.00            # Dividend yield
market_call = 0.1304 # Market call price
market_put = 0.1882  # Market put price

# ========== HESTON MODEL ==========
def heston_simulate(S0, v0, kappa, theta, sigma, rho, T, n_simulations=100000, n_steps=50):
    """Simulate Heston model paths with variance reduction."""
    dt = T / n_steps
    S = np.zeros((n_simulations, n_steps + 1))
    v = np.zeros_like(S)
    S[:, 0] = S0
    v[:, 0] = v0
    L = np.array([[1, 0], [rho, np.sqrt(1 - rho**2)]])
    
    # Antithetic variates for variance reduction
    Z = np.random.normal(size=(n_simulations//2, n_steps, 2))
    Z = np.concatenate([Z, -Z], axis=0)
    
    for t in range(1, n_steps + 1):
        W = Z[:, t-1, :] @ L.T
        v_prev = np.maximum(v[:, t-1], 1e-4)  # Ensure positive variance
        v[:, t] = np.maximum(v_prev + kappa*(theta - v_prev)*dt + 
                  sigma*np.sqrt(v_prev)*np.sqrt(dt)*W[:, 1], 1e-4)
        S[:, t] = S[:, t-1] * np.exp((r - q - 0.5*v_prev)*dt + 
                                   np.sqrt(v_prev)*np.sqrt(dt)*W[:, 0])
    return S, v

def heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    """Price options using Monte Carlo with antithetic variates."""
    S, _ = heston_simulate(S0, v0, kappa, theta, sigma, rho, T)
    payoff = np.maximum(S[:, -1] - K, 0) if option_type == 'call' else np.maximum(K - S[:, -1], 0)
    return np.exp(-r * T) * np.mean(payoff)

# ========== CALIBRATION ==========
def calibrate_heston():
    """Calibrate Heston parameters with relaxed but financially reasonable bounds."""
    def error(params):
        kappa, theta, sigma, rho, v0 = params
        try:
            call = heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, 'call')
            put = heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, 'put')
            err = 3.0 * (call - market_call)**2 + (put - market_put)**2  # Balanced weights
            print(f"Params: {np.round(params, 4)} | Call: {call:.4f} | Put: {put:.4f} | Error: {err:.6f}")
            return err
        except:
            return np.inf  # Return large error if simulation fails

    # Relaxed but financially reasonable bounds
    bounds = [
        (0.1, 10.0),     # kappa (mean reversion speed)
        (0.05, 1.0),      # theta (long-term variance)
        (0.05, 2.0),      # sigma (vol of vol)
        (-0.99, -0.1),    # rho (correlation)
        (0.01, 1.0)       # v0 (initial variance)
    ]
    
    # Reasonable initial guess
    initial_guess = [1.5, 0.3, 0.5, -0.7, 0.2]
    
    print("Calibrating... (This may take 2-5 minutes)")
    result = minimize(error, initial_guess, bounds=bounds, method='L-BFGS-B',
                     options={'maxiter': 100, 'ftol': 1e-6})

    if result.success:
        kappa, theta, sigma, rho, v0 = result.x
        print("\n=== Calibrated Parameters ===")
        print(f"kappa: {kappa:.4f} (mean reversion speed)")
        print(f"theta: {theta:.4f} (long-term variance)")
        print(f"sigma: {sigma:.4f} (vol of vol)")
        print(f"rho: {rho:.4f} (correlation)")
        print(f"v0: {v0:.4f} (initial variance)")
        print(f"\nImplied volatilities:")
        print(f"Short-term: {np.sqrt(v0):.2%}")
        print(f"Long-term: {np.sqrt(theta):.2%}")
        return result.x
    else:
        print("Calibration failed! Try adjusting initial guess.")
        return None

# ========== MAIN ==========
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    params = calibrate_heston()
    if params is None:
        exit()

    kappa, theta, sigma, rho, v0 = params

    # Price with calibrated parameters
    heston_call = heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, 'call')
    heston_put = heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, 'put')
    
    print("\n=== Market vs Heston Prices ===")
    print(f"{'Call Price':<15} Market: {market_call:.4f} | Heston: {heston_call:.4f}")
    print(f"{'Put Price':<15} Market: {market_put:.4f} | Heston: {heston_put:.4f}")

    # Greeks calculation with smaller bump
    def delta(S0, h=0.001):
        up = heston_price(S0 + h, K, T, r, kappa, theta, sigma, rho, v0, 'call')
        down = heston_price(S0 - h, K, T, r, kappa, theta, sigma, rho, v0, 'call')
        return (up - down) / (2 * h)
    
    print(f"\nDelta: {delta(S0):.4f}")

    # Simulate and plot volatility paths
    print("\nSimulating volatility paths...")
    _, v = heston_simulate(S0, v0, kappa, theta, sigma, rho, T, n_simulations=20)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.sqrt(v.T), alpha=0.75)
    plt.title("Simulated Volatility Paths (Heston Model)")
    plt.xlabel("Time Steps")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.show()