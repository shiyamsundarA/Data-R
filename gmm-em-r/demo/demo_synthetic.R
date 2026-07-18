# ============================================================
# demo_synthetic.R
#
# Generates synthetic 2D data from 3 known Gaussian components,
# fits the from-scratch vectorized GMM-EM algorithm, and
# visualizes convergence + final clustering with base R graphics.
# ============================================================

source(file.path("R", "gmm_em.R"))

set.seed(42)

# ---- 1. Simulate data from a known 3-component GMM (no MASS needed) ----
# Sample from a multivariate normal via its own Cholesky factor, so the
# whole demo stays dependency-free, matching the "from scratch" spirit.
rmvnorm_manual <- function(n, mu, Sigma) {
  d <- length(mu)
  L <- t(chol(Sigma))                     # Sigma = L L'
  Z <- matrix(rnorm(n * d), nrow = d)      # d x n standard normal
  X <- mu + L %*% Z                        # d x n
  t(X)
}

true_mu <- list(c(0, 0), c(5, 5), c(0, 6))
true_Sigma <- list(
  matrix(c(1, 0.3, 0.3, 1), 2, 2),
  matrix(c(1.2, -0.4, -0.4, 0.8), 2, 2),
  matrix(c(0.7, 0, 0, 1.5), 2, 2)
)
true_n <- c(300, 250, 200)

X <- do.call(rbind, Map(rmvnorm_manual, true_n, true_mu, true_Sigma))
true_labels <- rep(1:3, times = true_n)

# ---- 2. Fit GMM with from-scratch EM (best of several random restarts) ----
fit <- em_gmm_restarts(X, K = 3, n_restarts = 8, tol = 1e-8, max_iter = 300, seed = 1)

cat("\n=== Fit summary ===\n")
cat(sprintf("Converged: %s | Iterations: %d | Monotonic likelihood: %s\n",
            fit$converged, fit$iterations, fit$monotonic))
cat(sprintf("Final log-likelihood: %.4f\n", fit$loglik))

cat("\nEstimated mixture weights:\n")
print(round(fit$params$pi, 3))

cat("\nEstimated means:\n")
print(do.call(rbind, fit$params$mu))

cat("\nCross-tab of true vs. estimated cluster labels:\n")
print(table(true = true_labels, estimated = fit$labels))

# ---- 3. Plots ----
dir.create("output", showWarnings = FALSE)

# (a) Log-likelihood convergence curve
png("output/loglik_convergence.png", width = 700, height = 500)
plot(fit$loglik_history, type = "o", pch = 16, col = "steelblue",
     xlab = "EM iteration", ylab = "Log-likelihood",
     main = "GMM-EM Log-Likelihood Convergence")
grid()
dev.off()

# (b) Clustering result with estimated Gaussian contours
png("output/gmm_clusters.png", width = 700, height = 600)
palette_cols <- c("firebrick", "forestgreen", "royalblue")
plot(X, col = palette_cols[fit$labels], pch = 19, cex = 0.6,
     xlab = "x1", ylab = "x2",
     main = "GMM-EM Clustering (from scratch, vectorized)")

grid_seq <- seq(min(X[,1]) - 1, max(X[,1]) + 1, length.out = 150)
grid_seq2 <- seq(min(X[,2]) - 1, max(X[,2]) + 1, length.out = 150)
for (k in seq_len(fit$params$K)) {
  dens_grid <- outer(grid_seq, grid_seq2, Vectorize(function(a, b) {
    dmvnorm_vec(matrix(c(a, b), nrow = 1), fit$params$mu[[k]], fit$params$Sigma[[k]])
  }))
  contour(grid_seq, grid_seq2, dens_grid, add = TRUE, col = palette_cols[k], lwd = 1.5, nlevels = 4)
  points(fit$params$mu[[k]][1], fit$params$mu[[k]][2], pch = 4, cex = 2, lwd = 3, col = "black")
}
dev.off()

cat("\nPlots written to output/loglik_convergence.png and output/gmm_clusters.png\n")
