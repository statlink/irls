#include <RcppEigen.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::export]]
Rcpp::List gamma_cpp(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                     double tol = 1e-8, int maxiter = 100) {

  const int n = X.rows();
  const int p = X.cols() + 1;

  MatrixXd X_des(n, p);
  X_des.col(0).setOnes();
  X_des.rightCols(p - 1) = X;

  VectorXd beta(p), eta(n), mu(n), z(n);

  MatrixXd XtX = X_des.transpose() * X_des;
  Eigen::LDLT<MatrixXd> ldlt(XtX);
  MatrixXd XtX_inv = ldlt.solve(MatrixXd::Identity(p, p));

  for (int i = 0; i < n; ++i) {
    eta(i) = std::log(y(i));
  }

  beta.noalias() = XtX_inv * (X_des.transpose() * eta);
  eta.noalias() = X_des * beta;

  for (int i = 0; i < n; ++i) {
    mu(i) = std::exp(eta(i));
  }

  double dev_old = 0.0;
  for (int i = 0; i < n; ++i) {
    double ratio = y(i) / mu(i);
    dev_old += 2.0 * (ratio - 1 - std::log(ratio));
  }

  int iter = 0;
  double dev = dev_old;

  for (iter = 1; iter <= maxiter; ++iter) {

    for (int i = 0; i < n; ++i) {
      z(i) = eta(i) - 1 + y(i) / mu(i);
    }

    beta.noalias() = XtX_inv * (X_des.transpose() * z);

    eta.noalias() = X_des * beta;

    for (int i = 0; i < n; ++i) {
      mu(i) = std::exp(eta(i));
    }

    dev = 0.0;
    for (int i = 0; i < n; ++i) {
      double ratio = y(i) / mu(i);
      dev += 2.0 * ((y(i) - mu(i)) / mu(i) - std::log(ratio));
    }

    double rel_change = std::abs(dev - dev_old) / (0.1 + std::abs(dev));
    if (rel_change < tol) {
      break;
    }

    dev_old = dev;
  }

  double pearson = 0.0;
  for (int i = 0; i < n; ++i) {
    double resid = y(i) / mu(i) - 1;
    pearson += resid * resid;
  }

  double phi = pearson / (n - p);

  MatrixXd vcov = phi * XtX_inv;

  VectorXd se = vcov.diagonal().array().sqrt();

  return Rcpp::List::create(
      Rcpp::Named("coefficients") = beta, Rcpp::Named("deviance") = dev,
      Rcpp::Named("phi") = phi, Rcpp::Named("vcov") = vcov,
      Rcpp::Named("se") = se, Rcpp::Named("iterations") = iter);
}
