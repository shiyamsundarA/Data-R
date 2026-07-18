# ============================================================
# demo_iris.R
#
# Applies the from-scratch vectorized GMM-EM algorithm to the
# built-in `iris` dataset (4 numeric features) and compares the
# discovered clusters against the true species labels.
# No clustering library is used anywhere -- only base R.
# ============================================================

source(file.path("R", "gmm_em.R"))

data(iris)
X <- as.matrix(iris[, 1:4])
true_species <- iris$Species

fit <- em_gmm_restarts(X, K = 3, n_restarts = 10, tol = 1e-8, max_iter = 300, seed = 7)

cat("\n=== Iris GMM-EM fit summary ===\n")
cat(sprintf("Converged: %s | Iterations: %d | Monotonic likelihood: %s\n",
            fit$converged, fit$iterations, fit$monotonic))
cat(sprintf("Final log-likelihood: %.4f\n", fit$loglik))

cat("\nEstimated mixture weights:\n")
print(round(fit$params$pi, 3))

cat("\nConfusion matrix (true species vs. discovered cluster):\n")
print(table(species = true_species, cluster = fit$labels))
