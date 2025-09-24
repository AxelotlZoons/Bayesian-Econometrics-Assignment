# bayes_assignment_gibbs.py
# Bayesian regression with custom Gibbs sampler for:
#   log(sales_t) = gamma * (1 + beta1*display_t + beta2*coupon_t + beta3*log(price_t)) + eps_t
# with eps_t ~ N(0, sigma2)
# Priors: p(gamma) ∝ 1 (flat), beta_j ~ N(0, 2*sigma2), p(sigma2) ∝ 1/sigma2
#
# Usage (from a terminal or notebook):
#   python bayes_assignment_gibbs.py --brand 42
# where 42 should be replaced by the last two digits of your student number.
#
# The script:
#   - Loads sales.xls, price.xls, coupon.xls, display.xls from the working directory (or --data_dir)
#   - Selects the specified brand (1..100)
#   - Takes logs of sales and price (as required)
#   - Runs a Gibbs sampler (you can configure iterations/burn-in/thin)
#   - Saves posterior summaries to CSV and a quick PNG traceplot
#   - Prints key results to the console
#
# Notes:
#   - No external Bayesian packages are used; everything is coded from scratch.
#   - If your Excel engine has trouble with .xls, install `xlrd` or export to .xlsx once.
#
# © 2025 — Feel free to reuse for your coursework submission.

import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------

def inv_gamma_sample(alpha, beta, size=None, rng=None):
    """
    Sample from an inverse-gamma with shape alpha and scale beta
    using the relation: if G ~ Gamma(alpha, 1), then X = beta / G ~ IG(alpha, beta).
    """
    if rng is None:
        rng = np.random.default_rng()
    g = rng.gamma(shape=alpha, scale=1.0, size=size)
    return beta / g

def safe_log(x):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("Log requested of non-positive values. Check your data (sales and price must be > 0).")
    return np.log(x)

def read_dataframes(data_dir):
    paths = {
        'sales':   os.path.join(data_dir, 'sales.xls'),
        'price':   os.path.join(data_dir, 'price.xls'),
        'coupon':  os.path.join(data_dir, 'coupon.xls'),
        'display': os.path.join(data_dir, 'display.xls'),
    }
    dfs = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
        try:
            dfs[k] = pd.read_excel(p, header=0)
        except Exception as e:
            # Try engine openpyxl if .xlsx, or provide a hint
            raise RuntimeError(f"Failed to read {p}. If it's .xls, ensure 'xlrd' is available. Original error: {e}")
    return dfs

def select_brand_column(df, brand_idx):
    """
    Each Excel is assumed to have 100 brand columns.
    We accept either a 'BrandXX' style header or plain numeric columns.
    The brand_idx is 1..100.
    """
    # Try exact matching first
    colnames = list(df.columns)
    candidates = []
    # typical patterns like 'Brand1', 'Brand01', '1', 1, etc.
    patterns = {f"Brand{brand_idx}", f"Brand{brand_idx:02d}", str(brand_idx), brand_idx}
    for c in colnames:
        if c in patterns:
            return df[c].to_numpy()
        candidates.append(c)

    # If no direct match, assume columns are ordered by brand 1..100, ignoring the first column if it's 'Week' etc.
    # Try to detect a non-numeric first column (like 'Week'), then count 100 brand columns after it.
    numeric_cols = [c for c in colnames if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 100:
        # assume the first 100 numeric columns are brands 1..100
        # but ensure we keep the original order
        ordered_numeric_cols = [c for c in colnames if c in numeric_cols]
        return df[ordered_numeric_cols[brand_idx-1]].to_numpy()

    # Fallback: just take the brand_idx-th column after dropping any obvious non-brand columns
    drop_like = {'week', 'weeks', 'time', 't'}
    filtered = [c for c in colnames if str(c).strip().lower() not in drop_like]
    if len(filtered) < brand_idx:
        raise ValueError("Could not locate the requested brand column. Inspect your Excel headers.")
    return df[filtered[brand_idx-1]].to_numpy()

def build_design(coupon, display, price_log):
    """
    Construct X with columns [display, coupon, log(price)], in that order.
    """
    X = np.column_stack([display, coupon, price_log])
    return X

def gibbs_sampler(y, X, n_iter=20000, burn=5000, thin=5, seed=2025,
                  sigma2_init=1.0, gamma_init=1.0, verbose=True):
    """
    Gibbs sampler for:
        y = gamma * (1 + X beta) + eps,  eps ~ N(0, sigma2 I)
    Priors:
        p(gamma) ∝ 1
        beta ~ N(0, 2 sigma2 I)
        p(sigma2) ∝ 1/sigma2
    Returns draws for gamma, beta (p=3), sigma2 and derived theta = gamma * beta.
    """
    rng = np.random.default_rng(seed)
    T, p = X.shape

    # Storage sizes
    kept = (n_iter - burn) // thin
    beta_draws  = np.zeros((kept, p))
    gamma_draws = np.zeros(kept)
    sig2_draws  = np.zeros(kept)
    theta_draws = np.zeros((kept, p))  # theta_j = gamma * beta_j

    # Initialize
    sigma2 = float(sigma2_init)
    gamma  = float(gamma_init)
    beta   = np.zeros(p)

    one = np.ones(T)

    keep_idx = 0
    for it in range(1, n_iter + 1):
        # --- [1] beta | gamma, sigma2, y  ~ N(mb, Vb)
        # y - gamma*1 = gamma * X beta + eps
        ytil = y - gamma * one
        # Posterior: Vb = sigma2 * (gamma^2 X'X + 1/2 I)^(-1)
        XtX = X.T @ X
        A = gamma**2 * XtX + 0.5 * np.eye(p)
        try:
            A_chol = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # add tiny ridge for stability
            A_chol = np.linalg.cholesky(A + 1e-10 * np.eye(p))
        # Solve A * mb = gamma * X' ytil
        rhs = gamma * (X.T @ ytil)
        mb = np.linalg.solve(A, rhs)
        # Draw beta = mb + sqrt(sigma2) * A^{-1/2} * z
        z = rng.normal(size=p)
        # Solve A^{1/2} u = z  -> u = A^{-1/2} z using chol
        u = np.linalg.solve(A_chol, z)
        u = np.linalg.solve(A_chol.T, u)
        beta = mb + math.sqrt(sigma2) * u

        # --- [2] gamma | beta, sigma2, y ~ N(mg, Vg)
        zvec = one + X @ beta
        zz = float(zvec @ zvec)
        zy = float(zvec @ y)
        Vg = sigma2 / zz
        mg = zy / zz
        gamma = rng.normal(loc=mg, scale=math.sqrt(Vg))

        # --- [3] sigma2 | beta, gamma, y ~ IG(alpha, beta_scale)
        resid = y - gamma * zvec
        SSE = float(resid @ resid)
        # prior on beta contributes (beta' beta) / (4*sigma2) in exponential form
        alpha = 0.5 * (T + p)
        beta_scale = 0.5 * (SSE + 0.5 * float(beta @ beta))
        sigma2 = inv_gamma_sample(alpha, beta_scale, rng=rng)

        # Save draws
        if it > burn and ((it - burn) % thin == 0):
            gamma_draws[keep_idx] = gamma
            sig2_draws[keep_idx] = sigma2
            beta_draws[keep_idx, :] = beta
            theta_draws[keep_idx, :] = gamma * beta
            keep_idx += 1

        if verbose and (it % max(1000, thin) == 0):
            print(f"Iter {it:6d} | gamma={gamma: .4f}  sigma2={sigma2: .4f}  beta={beta}")

    draws = {
        'gamma': gamma_draws,
        'sigma2': sig2_draws,
        'beta': beta_draws,
        'theta': theta_draws,  # implied coefficients on [display, coupon, log(price)]
    }
    return draws

def summarize_draws(draws, names):
    arr = np.column_stack([np.mean(draws, axis=0),
                           np.std(draws, axis=0, ddof=1),
                           np.quantile(draws, 0.025, axis=0),
                           np.quantile(draws, 0.975, axis=0)])
    df = pd.DataFrame(arr, columns=['mean', 'sd', 'q2.5', 'q97.5'], index=names)
    return df

def main():
    parser = argparse.ArgumentParser(description="Bayesian Gibbs sampler for log-sales regression with promotions and price.")
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing sales.xls, price.xls, coupon.xls, display.xls')
    parser.add_argument('--brand', type=int, required=True, help='Brand index: last two digits of your student number (1..100)')
    parser.add_argument('--iters', type=int, default=20000, help='Total iterations')
    parser.add_argument('--burn', type=int, default=5000, help='Burn-in iterations')
    parser.add_argument('--thin', type=int, default=5, help='Thinning interval')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--sigma2_init', type=float, default=1.0, help='Initial sigma^2')
    parser.add_argument('--gamma_init', type=float, default=1.0, help='Initial gamma')
    parser.add_argument('--no_plots', action='store_true', help='Skip creating trace plots')
    args = parser.parse_args()

    if not (1 <= args.brand <= 100):
        raise ValueError("Brand must be in 1..100. Use the last two digits of your student number.")

    dfs = read_dataframes(args.data_dir)
    # Extract series for the chosen brand
    sales  = select_brand_column(dfs['sales'],   args.brand)
    price  = select_brand_column(dfs['price'],   args.brand)
    coupon = select_brand_column(dfs['coupon'],  args.brand)
    display= select_brand_column(dfs['display'], args.brand)

    # Check lengths match
    n = len(sales)
    for v, name in [(price, 'price'), (coupon, 'coupon'), (display, 'display')]:
        if len(v) != n:
            raise ValueError(f"Length mismatch: sales has {n}, but {name} has {len(v)}")

    # Required: take logs of sales and price
    y = safe_log(sales)
    lp = safe_log(price)

    # Build design matrix X = [display, coupon, log(price)]
    X = build_design(coupon=coupon, display=display, price_log=lp)

    # Run Gibbs
    draws = gibbs_sampler(y, X,
                          n_iter=args.iters,
                          burn=args.burn,
                          thin=args.thin,
                          seed=args.seed,
                          sigma2_init=args.sigma2_init,
                          gamma_init=args.gamma_init,
                          verbose=True)

    # Summaries
    beta_names = ['beta_display', 'beta_coupon', 'beta_logprice']
    theta_names = ['theta_display', 'theta_coupon', 'theta_logprice']

    summ_gamma = summarize_draws(draws['gamma'][:, None], ['gamma'])
    summ_sigma2= summarize_draws(draws['sigma2'][:, None], ['sigma2'])
    summ_beta  = summarize_draws(draws['beta'], beta_names)
    summ_theta = summarize_draws(draws['theta'], theta_names)

    # Save outputs
    outdir = os.path.join(args.data_dir, f'brand_{args.brand:02d}_output')
    os.makedirs(outdir, exist_ok=True)
    summ_gamma.to_csv(os.path.join(outdir, 'summary_gamma.csv'))
    summ_sigma2.to_csv(os.path.join(outdir, 'summary_sigma2.csv'))
    summ_beta.to_csv(os.path.join(outdir, 'summary_beta.csv'))
    summ_theta.to_csv(os.path.join(outdir, 'summary_theta.csv'))

    # Basic trace plots (optional)
    if not args.no_plots:
        # gamma
        plt.figure()
        plt.plot(draws['gamma'])
        plt.title('Trace: gamma')
        plt.xlabel('draw')
        plt.ylabel('gamma')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'trace_gamma.png'), dpi=150)
        plt.close()

        # sigma2
        plt.figure()
        plt.plot(draws['sigma2'])
        plt.title('Trace: sigma^2')
        plt.xlabel('draw')
        plt.ylabel('sigma^2')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'trace_sigma2.png'), dpi=150)
        plt.close()

        # beta
        for j, name in enumerate(beta_names):
            plt.figure()
            plt.plot(draws['beta'][:, j])
            plt.title(f'Trace: {name}')
            plt.xlabel('draw')
            plt.ylabel(name)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'trace_{name}.png'), dpi=150)
            plt.close()

        # theta
        for j, name in enumerate(theta_names):
            plt.figure()
            plt.plot(draws['theta'][:, j])
            plt.title(f'Trace: {name} (gamma*beta)')
            plt.xlabel('draw')
            plt.ylabel(name)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'trace_{name}.png'), dpi=150)
            plt.close()

    # Console print
    print("\nPosterior summaries (means, sds, 95% intervals):")
    print("\nGamma:\n", summ_gamma)
    print("\nSigma^2:\n", summ_sigma2)
    print("\nBetas (on display, coupon, log(price)):\n", summ_beta)
    print("\nThetas = gamma * betas (implied coefficients in expanded linear form):\n", summ_theta)
    print(f"\nFiles saved in: {outdir}")

if __name__ == '__main__':
    main()
