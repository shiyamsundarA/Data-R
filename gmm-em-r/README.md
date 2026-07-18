# Vectorized EM Algorithm for Gaussian Mixture Models (GMM)

A from-scratch, fully vectorized implementation of the Expectation-Maximization
(EM) algorithm for Gaussian Mixture Models (GMMs) in base R.

**No clustering package is used anywhere** (no `mclust`, `ClusterR`,
`EMCluster`, `stats::kmeans`, etc.). Multivariate normal densities,
posterior responsibilities, parameter updates, and log-likelihood
convergence checks are all implemented directly with base R matrix algebra.

## What's implemented

| Requirement | Where |
|---|---|
| Vectorized multivariate normal density | `dmvnorm_vec()` — one Cholesky factorization + matrix solve per component, no per-point loop |
| Dynamic posterior probabilities (E-step) | `e_step()` — responsibility matrix `r[i,k]` computed for all points and components at once |
| Mixture weight / mean / covariance updates (M-step) | `m_step()` — weighted mean via `crossprod`, weighted covariance via a single `crossprod` per component |
| Log-likelihood convergence verification | `em_gmm()` — tracks the log-likelihood every iteration, checks the theoretical **non-decreasing** guarantee of EM, and stops once the relative change drops below `tol` |
| No clustering library | Initialization uses random data points + the global sample covariance, not k-means or any packaged clustering routine |

## Project structure

```
gmm-em-r/
├── R/
│   └── gmm_em.R          # core algorithm: dmvnorm_vec, e_step, m_step, em_gmm, em_gmm_restarts
├── demo/
│   ├── demo_synthetic.R  # 3-component synthetic 2D example + convergence/cluster plots
│   └── demo_iris.R       # applies the algorithm to the built-in iris dataset
├── output/                # generated plots (created when you run the demo)
├── .gitignore
├── LICENSE
└── README.md
```

## The math

**Model.** Data `x_1, ..., x_n` are assumed drawn from a mixture of `K`
multivariate Gaussians:

```
p(x) = sum_{k=1}^{K} pi_k * N(x ; mu_k, Sigma_k)
```

**E-step.** Given current parameters, compute the responsibility of
component `k` for point `i`:

```
r[i,k] = pi_k * N(x_i ; mu_k, Sigma_k) / sum_j pi_j * N(x_i ; mu_j, Sigma_j)
```

**M-step.** Update parameters using the responsibilities:

```
N_k     = sum_i r[i,k]
pi_k    = N_k / n
mu_k    = (1 / N_k) * sum_i r[i,k] * x_i
Sigma_k = (1 / N_k) * sum_i r[i,k] * (x_i - mu_k)(x_i - mu_k)'
```

**Convergence.** EM theory guarantees the observed-data log-likelihood

```
LL = sum_i log( sum_k pi_k * N(x_i ; mu_k, Sigma_k) )
```

never decreases between iterations. `em_gmm()` checks this guarantee at
every step (flagging any drop beyond floating-point noise as a possible
bug) and stops once `|LL_new - LL_old| / |LL_old| < tol`.

## Usage

```r
source("R/gmm_em.R")

# X: an n x d numeric matrix or data.frame
fit <- em_gmm_restarts(X, K = 3, n_restarts = 8, tol = 1e-8, seed = 1)

fit$params$pi      # estimated mixture weights
fit$params$mu      # estimated mean vectors (list of length K)
fit$params$Sigma   # estimated covariance matrices (list of length K)
fit$labels         # hard cluster assignment per point (argmax responsibility)
fit$loglik_history # log-likelihood at every iteration
fit$monotonic      # TRUE if the non-decreasing guarantee held throughout
```

Run the demos:

```bash
cd gmm-em-r
Rscript demo/demo_synthetic.R
Rscript demo/demo_iris.R
```

The synthetic demo writes two plots to `output/`:
- `loglik_convergence.png` — log-likelihood vs. iteration
- `gmm_clusters.png` — final clustering with estimated Gaussian contours


Base R only (tested on R 4.3). No external packages are required to run
`R/gmm_em.R` itself.
