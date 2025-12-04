col.irls <- function(y, x, type = "logistic", maxiter = 100, tol = 1e-6, parallel = FALSE) {
  if ( type == "logistic" ) {
    res <- logis_new(x, y, maxiter = maxiter, tol = tol, parallel = parallel)
    colnames(res) <- c("alpha", "beta", "deviance")
  } else if ( type == "poisson" ) {
    res <- poiss_new(x, y, maxiter = maxiter, tol = tol, parallel = parallel)
    colnames(res) <- c("alpha", "beta", "deviance")
  } else if ( type == "gamma" )  {
    res <- gammas_new(x, y, maxiter = maxiter, tol = tol, parallel = parallel)
    colnames(res) <- c("alpha", "beta", "deviance", "phi")
  }
  res
}
