# ============================================================
# gmm_em.R
#
# Vectorized Expectation-Maximization (EM) algorithm for
# Gaussian Mixture Models (GMM), implemented from scratch in
# base R. No clustering package (mclust, ClusterR, EMCluster,
# stats::kmeans, etc.) is used anywhere in this file.
#
# Author: (your name here)
# ============================================================

# ------------------------------------------------------------
# 1. Vectorized multivariate normal density
# ------------------------------------------------------------
# Evaluates N(x ; mu, Sigma) for EVERY row of X at once using a
# single Cholesky factorization + matrix solve, instead of
# looping row-by-row. This is the "vectorized" core of the
# whole algorithm.
#
#   X     : n x d data matrix
#   mu    : length-d mean vector
#   Sigma : d x d covariance matrix
#
# Returns: length-n numeric vector of densities.
dmvnorm_vec <- function(X, mu, Sigma) {
  X <- as.matrix(X)
  d <- ncol(X)
  n <- nrow(X)

  # Cholesky factorization: Sigma = L %*% t(L), L lower-triangular.
  # chol() in base R returns the upper factor U s.t. Sigma = t(U) %*% U,
  # so we work with U directly (U = t(L)).
  U <- tryCatch(
    chol(Sigma),
    error = function(e) {
      # Fall back to a tiny ridge if Sigma is numerically singular
      chol(Sigma + diag(1e-6, d))
    }
  )

  # Center all rows of X by mu at once (matrix - vector, recycled by column)
  Xc <- sweep(X, 2, mu, FUN = "-")   # n x d

  # Solve U' Z' = Xc'  <=>  Z = Xc %*% solve(U)  but done via
  # forwardsolve/backsolve on the triangular factor for speed & stability.
  # backsolve(U, ., transpose = TRUE) solves t(U) %*% Y = t(Xc)
  Z <- backsolve(U, t(Xc), transpose = TRUE)   # d x n

  # Squared Mahalanobis distance for every point, vectorized: colSums(Z^2)
  mahal_sq <- colSums(Z^2)                     # length n

  log_det_Sigma <- 2 * sum(log(diag(U)))
  log_dens <- -0.5 * (d * log(2 * pi) + log_det_Sigma + mahal_sq)

  exp(log_dens)
}


# ------------------------------------------------------------
# 2. Parameter initialization (NO clustering library used)
# ------------------------------------------------------------
# Means are initialized from K distinct randomly-chosen data
# points; covariances start at the (regularized) sample
# covariance of the full data set; weights start uniform.
init_params <- function(X, K, seed = NULL) {
  X <- as.matrix(X)
  n <- nrow(X); d <- ncol(X)
  if (!is.null(seed)) set.seed(seed)

  idx <- sample.int(n, K)
  mu_list <- lapply(idx, function(i) X[i, ])

  S0 <- cov(X) + diag(1e-6, d)  # global sample covariance, regularized
  Sigma_list <- replicate(K, S0, simplify = FALSE)

  pi_k <- rep(1 / K, K)

  list(pi = pi_k, mu = mu_list, Sigma = Sigma_list, K = K, d = d)
}


# ------------------------------------------------------------
# 3. E-step: dynamic posterior responsibilities
# ------------------------------------------------------------
# Computes the n x K responsibility matrix
#   r[i,k] = pi_k * N(x_i ; mu_k, Sigma_k) / sum_j pi_j * N(x_i ; mu_j, Sigma_j)
# fully vectorized: one dmvnorm_vec() call per component (K calls
# total, never a call per data point), then a single row-normalization.
e_step <- function(X, params) {
  n <- nrow(X); K <- params$K
  dens <- matrix(0, nrow = n, ncol = K)

  for (k in seq_len(K)) {
    dens[, k] <- params$pi[k] * dmvnorm_vec(X, params$mu[[k]], params$Sigma[[k]])
  }

  row_sums <- rowSums(dens)
  row_sums[row_sums <= 0] <- .Machine$double.eps  # numerical safety

  resp <- dens / row_sums          # vectorized row-wise normalization
  list(resp = resp, dens = dens, row_sums = row_sums)
}


# ------------------------------------------------------------
# 4. M-step: update weights, means, covariances
# ------------------------------------------------------------
# All updates are matrix expressions -- no per-point loops.
#   N_k     = sum_i r[i,k]
#   pi_k    = N_k / n
#   mu_k    = (1/N_k) * sum_i r[i,k] * x_i
#   Sigma_k = (1/N_k) * sum_i r[i,k] * (x_i - mu_k)(x_i - mu_k)'
m_step <- function(X, resp, reg = 1e-6) {
  X <- as.matrix(X)
  n <- nrow(X); d <- ncol(X); K <- ncol(resp)

  Nk <- colSums(resp)                       # length K
  Nk_safe <- pmax(Nk, .Machine$double.eps)

  pi_k <- Nk / n

  # Weighted means: (K x d) = t(resp) %*% X, each row divided by Nk
  mu_mat <- (t(resp) %*% X) / Nk_safe        # K x d matrix
  mu_list <- lapply(seq_len(K), function(k) mu_mat[k, ])

  Sigma_list <- vector("list", K)
  for (k in seq_len(K)) {
    Xc <- sweep(X, 2, mu_list[[k]], FUN = "-")     # n x d, centered
    w  <- resp[, k]                                # length n weights
    # Weighted scatter matrix via a single crossprod call (vectorized
    # over all n points at once): sum_i w_i * Xc_i Xc_i' = t(Xc*w) %*% Xc
    Sigma_k <- crossprod(Xc * w, Xc) / Nk_safe[k]
    Sigma_list[[k]] <- Sigma_k + diag(reg, d)      # ridge for stability
  }

  list(pi = pi_k, mu = mu_list, Sigma = Sigma_list, K = K, d = d)
}


# ------------------------------------------------------------
# 5. Observed-data log-likelihood
# ------------------------------------------------------------
#   loglik = sum_i log( sum_k pi_k * N(x_i ; mu_k, Sigma_k) )
loglik_gmm <- function(X, params) {
  e <- e_step(X, params)
  sum(log(e$row_sums))
}


# ------------------------------------------------------------
# 6. Main EM driver with mathematical convergence verification
# ------------------------------------------------------------
# EM theory guarantees the observed-data log-likelihood is
# non-decreasing at every iteration. This function:
#   (a) iterates E-step -> M-step,
#   (b) records the log-likelihood at every iteration,
#   (c) checks the monotonicity guarantee at each step and warns
#       if it is violated beyond floating-point tolerance
#       (a real violation would indicate a bug),
#   (d) stops when the *relative* increase in log-likelihood
#       drops below `tol`, which is the standard convergence
#       constraint |LL_new - LL_old| / |LL_old| < tol.
em_gmm <- function(X, K, tol = 1e-6, max_iter = 500,
                    seed = NULL, verbose = TRUE, reg = 1e-6) {
  X <- as.matrix(X)
  params <- init_params(X, K, seed = seed)

  ll_history <- numeric(max_iter)
  prev_ll <- -Inf
  converged <- FALSE
  monotonic_ok <- TRUE

  for (it in seq_len(max_iter)) {
    e <- e_step(X, params)
    cur_ll <- sum(log(e$row_sums))
    ll_history[it] <- cur_ll

    # --- mathematical verification of the EM monotonicity guarantee ---
    # A genuine EM implementation cannot decrease the log-likelihood; only
    # flag drops that exceed ordinary floating-point noise.
    noise_tol <- 1e-6 * max(abs(prev_ll), 1)
    if (it > 1 && (cur_ll - prev_ll) < -noise_tol) {
      monotonic_ok <- FALSE
      warning(sprintf(
        "Log-likelihood DECREASED at iteration %d (%.6f -> %.6f). Check implementation / regularization.",
        it, prev_ll, cur_ll
      ))
    }

    if (verbose) {
      cat(sprintf("iter %3d | log-likelihood = %.6f\n", it, cur_ll))
    }

    # --- convergence constraint ---
    rel_change <- abs(cur_ll - prev_ll) / max(abs(prev_ll), 1e-8)
    if (it > 1 && rel_change < tol) {
      converged <- TRUE
      prev_ll <- cur_ll
      params <- m_step(X, e$resp, reg = reg)  # final parameter update
      break
    }

    params <- m_step(X, e$resp, reg = reg)
    prev_ll <- cur_ll
  }

  ll_history <- ll_history[1:it]
  final_resp <- e_step(X, params)$resp
  labels <- apply(final_resp, 1, which.max)

  list(
    params = params,
    resp = final_resp,
    labels = labels,
    loglik = prev_ll,
    loglik_history = ll_history,
    iterations = it,
    converged = converged,
    monotonic = monotonic_ok
  )
}


# ------------------------------------------------------------
# 7. Convenience: fit with multiple random restarts
# ------------------------------------------------------------
# Since initialization is random (no clustering library used to
# seed it), running several restarts and keeping the best final
# log-likelihood is standard practice for avoiding poor local optima.
em_gmm_restarts <- function(X, K, n_restarts = 5, tol = 1e-6,
                             max_iter = 500, seed = NULL, reg = 1e-6) {
  best <- NULL
  for (r in seq_len(n_restarts)) {
    s <- if (is.null(seed)) NULL else seed + r
    fit <- em_gmm(X, K, tol = tol, max_iter = max_iter,
                  seed = s, verbose = FALSE, reg = reg)
    if (is.null(best) || fit$loglik > best$loglik) best <- fit
  }
  best
}
