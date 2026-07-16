# ==============================
# EM Algorithm for GMM from Scratch
# ==============================

set.seed(123)

# ------------------------------
# Sample Dataset
# ------------------------------
data1 <- matrix(rnorm(100, mean = 2, sd = 1), ncol = 2)
data2 <- matrix(rnorm(100, mean = 7, sd = 1), ncol = 2)
X <- rbind(data1, data2)

n <- nrow(X)
d <- ncol(X)
K <- 2

# ------------------------------
# Initialize Parameters
# ------------------------------
weights <- rep(1/K, K)
means <- X[sample(1:n, K), ]
covs <- array(0, dim = c(d, d, K))

for(k in 1:K){
  covs[,,k] <- diag(d)
}

# ------------------------------
# Multivariate Gaussian Function
# ------------------------------
dmvnorm_custom <- function(x, mean, sigma){

  sigma <- sigma + diag(1e-6, nrow(sigma))

  detSigma <- det(sigma)

  invSigma <- solve(sigma)

  diff <- matrix(x - mean, ncol = 1)

  exponent <- -0.5 * t(diff) %*% invSigma %*% diff

  value <- exp(exponent) /
    sqrt((2*pi)^length(mean) * detSigma)

  return(as.numeric(value))
}

# ------------------------------
# EM Parameters
# ------------------------------
max_iter <- 100
tol <- 1e-6
loglik_old <- -Inf

# ------------------------------
# EM Algorithm
# ------------------------------
for(iter in 1:max_iter){

  # ---------- E Step ----------
  gamma <- matrix(0, n, K)

  for(i in 1:n){

    for(k in 1:K){

      gamma[i,k] <- weights[k] *
        dmvnorm_custom(X[i,], means[k,], covs[,,k])

    }

    gamma[i,] <- gamma[i,] / sum(gamma[i,])

  }

  # ---------- M Step ----------
  Nk <- colSums(gamma)

  weights <- Nk / n

  for(k in 1:K){

    means[k,] <- colSums(gamma[,k] * X) / Nk[k]

    sigma <- matrix(0, d, d)

    for(i in 1:n){

      diff <- matrix(X[i,] - means[k,], ncol=1)

      sigma <- sigma +
        gamma[i,k] * (diff %*% t(diff))

    }

    covs[,,k] <- sigma / Nk[k] +
      diag(1e-6, d)

  }

  # ---------- Log Likelihood ----------
  loglik <- 0

  for(i in 1:n){

    temp <- 0

    for(k in 1:K){

      temp <- temp +
        weights[k] *
        dmvnorm_custom(X[i,], means[k,], covs[,,k])

    }

    loglik <- loglik + log(temp)

  }

  cat("Iteration:", iter,
      " LogLikelihood:", loglik, "\n")

  if(abs(loglik - loglik_old) < tol){

    cat("\nAlgorithm Converged!\n")

    break

  }

  loglik_old <- loglik

}

# ------------------------------
# Final Cluster Assignment
# ------------------------------
clusters <- apply(gamma, 1, which.max)

# ------------------------------
# Results
# ------------------------------
cat("\nFinal Mixture Weights:\n")
print(weights)

cat("\nFinal Means:\n")
print(means)

cat("\nFinal Covariance Matrices:\n")
print(covs)

cat("\nCluster Assignment:\n")
print(clusters)

# ------------------------------
# Plot
# ------------------------------
plot(X,
     col = clusters,
     pch = 19,
     main = "Gaussian Mixture Model using EM Algorithm")
points(means,
       col = "black",
       pch = 8,
       cex = 2)