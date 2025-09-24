import os, math
import pandas as pd
import numpy as np

def load_brand(data_dir, digits):

    print(f"\nDataset used: ", digits)

    files = {
        "sales":   "sales.xls",
        "price":   "price.xls",
        "coupon":  "coupon.xls",
        "display": "displ.xls",
    }

    colname = f"brand{digits}"

    # read each file and extract the 'brand89' column
    cols = {}
    for key, file_name in files.items():
        path = os.path.join(data_dir, file_name)
        df = pd.read_excel(path)
        col_values = df[colname]
        cols[key] = col_values

    # align by row and assemble a single DataFrame
    df = pd.DataFrame({
        "sales": cols["sales"].to_numpy(),
        "price": cols["price"].to_numpy(),
        "coupon": cols["coupon"].to_numpy(),
        "display": cols["display"].to_numpy(),
    })

    return df


def make_y_X(df):

    # log-transform sales and price (No 0s in data)
    y  = np.log(df["sales"].to_numpy(float))
    price_logged = np.log(df["price"].to_numpy(float))
    X  = np.column_stack([df["display"].to_numpy(float), df["coupon"].to_numpy(float), price_logged])

    return y, X


def inverse_gamma_sample(alpha, beta, rng):
    
    # Draw X ~ InvGamma(alpha, beta)
    gamma_draws = rng.gamma(shape=alpha, scale=1)
    inverse_gamma_draws = beta / gamma_draws

    return inverse_gamma_draws

def gibbs_sample(y, X, iterations, burn, thin, rng):

    print(f"\nSimulations: ", iterations)
    print(f"Burn-in simulations: ", burn)
    print(f"Thin value: ", thin)

    # Gibbs sampler for the model
    T, p = X.shape
    keep = (iterations - burn) // thin

    beta_draws  = np.zeros((keep, p))
    gamma_draws = np.zeros(keep)
    sigma2_draws  = np.zeros(keep)

    # init
    sigma2 = 1.0
    gamma  = 1.0
    beta   = np.zeros(p)
    one    = np.ones(T)

    k = 0
    for iteration in range(1, iterations + 1):
        # beta | gamma, sigma2, y
        ytil = y - gamma * one
        XtX  = X.T @ X
        A    = (gamma**2) * XtX + 0.5 * np.eye(p)
        rhs  = gamma * (X.T @ ytil)
        mb   = np.linalg.solve(A, rhs)

        # draw beta = mb + sqrt(sigma2)*A^{-1/2} z  via Cholesky
        L = np.linalg.cholesky(A + 1e-12*np.eye(p))
        z = rng.normal(size=p)
        u = np.linalg.solve(L, z)
        u = np.linalg.solve(L.T, u)
        beta = mb + math.sqrt(sigma2) * u

        # gamma | beta, sigma2, y
        zvec = one + X @ beta
        zz   = float(zvec @ zvec)
        zy   = float(zvec @ y)
        Vg   = sigma2 / zz
        mg   = zy / zz
        gamma = rng.normal(loc=mg, scale=math.sqrt(Vg))

        # sigma^2 | gamma, beta, y
        resid = y - gamma * zvec
        SSE   = float(resid @ resid)
        alpha = 0.5 * (T + p)
        b     = 0.5 * (SSE + 0.5 * float(beta @ beta))
        sigma2 = float(inverse_gamma_sample(alpha, b, rng))

        # save
        if iteration > burn and ((iteration - burn) % thin == 0):
            gamma_draws[k] = gamma
            sigma2_draws[k] = sigma2
            beta_draws[k]  = beta
            k += 1

    return {"gamma": gamma_draws, "sigma2": sigma2_draws, "beta": beta_draws}


def sum_1d(a):

    a = np.asarray(a)
    quantiles = np.quantile(a, [0.1, 0.9])

    return {
        "quantile_10%": float(quantiles[0]),
        "mean":         float(a.mean()),
        "quantile_90%": float(quantiles[1]),
    }


def sum_2d(A):

    A = np.asarray(A)
    quantiles = np.quantile(A, [0.1, 0.9], axis=0)
    
    return {
        "quantile_10%": quantiles[0],
        "mean":         A.mean(axis=0),
        "quantile_90%": quantiles[1],
    }


def summarize_for_sheet(draws):

    beta_names  = ["beta_display","beta_coupon","beta_logprice"]

    res = {
        "gamma":      sum_1d(draws["gamma"]),
        "beta":       sum_2d(draws["beta"]),
        "beta_names": beta_names,
        "sigma2":     sum_1d(draws["sigma2"]),
    }
    return res


def table_from_res(res, decimals=4):

    # assemble a DataFrame for pretty printing
    cols = ["quantile_10%", "mean", "quantile_90%"]

    # 1-row frames for gamma and sigma2
    gamma_df = pd.DataFrame([{col: res["gamma"][col] for col in cols}], index=["gamma"])
    sigma2_df = pd.DataFrame([{col: res["sigma2"][col] for col in cols}], index=["sigma2"])

    # 3-row frame for betas (index are the beta names)
    beta_df = pd.DataFrame({
        "quantile_10%": res["beta"]["quantile_10%"],
        "mean":         res["beta"]["mean"],
        "quantile_90%": res["beta"]["quantile_90%"],
    }, index=res["beta_names"])

    table = pd.concat([gamma_df, beta_df, sigma2_df], axis=0)

    return table.round(decimals)


if __name__ == "__main__":

    SEED = 12345
    
    df = load_brand("data", digits="89")
    y, X = make_y_X(df)

    rng = np.random.default_rng(SEED)
    draws = gibbs_sample(y, X, iterations=100000, burn=10000, thin=5, rng=rng)

    # compute summaries to fill in the sheet
    res = summarize_for_sheet(draws)

    # pretty print
    table = table_from_res(res, decimals=4)
    print("\n",table)

    prob_neg = (draws["gamma"] * draws["beta"][:, 2] < 0).mean()
    print(f"\nP(effect logprice on logsales < 0) = {prob_neg:.4f}")