irls <- function(y, x, type = "logistic", maxiter = 100, tol = 1e-6) {
  x <- model.matrix( y ~., data = as.data.frame(x) )
  if ( type == "logistic" ) {
    res <- logistic_cpp(x, y, maxiter = maxiter, tol = tol)
  } else if ( type == "poisson" ) {
    res <- poisson_cpp(x, y, maxiter = maxiter, tol = tol)
  } else if ( type == "gamma" )  res <- gamma_cpp(x, y, maxiter = maxiter, tol = tol)
  res
}
