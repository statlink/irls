#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::export]]
Rcpp::List poisson_cpp(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                       double tol = 1e-6,

                       int maxiter = 100) {
  const int n = X.rows();
  const int p = X.cols() + 1;

  MatrixXd X_des(n, p);
  X_des.col(0).setOnes();
  X_des.rightCols(p - 1) = X;

  VectorXd beta = VectorXd::Zero(p);
  VectorXd eta(n), mu(n), w(n), z(n), XtWz(p), se(p);
  MatrixXd XtWX(p, p);

  const double my = y.mean();
  const double eta0 = std::log(my);
  eta.setConstant(eta0);
  mu.setConstant(std::exp(eta0));

  double dev1 = 0.0;
  for (int i = 0; i < n; ++i) {
    if (y(i) > 0) {
      dev1 += y(i) * std::log(y(i) / mu(i));
    }
    dev1 += mu(i) - y(i);
  }
  dev1 *= 2.0;

  w.array() = mu.array().max(1e-8);
  z.array() = eta0 - 1.0 + y.array() / my;

  MatrixXd Xt= X_des.transpose();

  XtWX.noalias() = Xt * w.asDiagonal() * X_des;
  XtWz.noalias() = Xt * (w.array() * z.array()).matrix();

  beta = XtWX.llt().solve(XtWz);
  eta.noalias() = X_des * beta;

  for (int i = 0; i < n; ++i) {
    mu(i) = std::exp(eta(i));
  }

  double dev2 = 0.0;
  for (int i = 0; i < n; ++i) {
    if (y(i) > 0) {
      dev2 += y(i) * std::log(y(i) / mu(i));
    }
    dev2 += mu(i) - y(i);
  }
  dev2 *= 2.0;

  int iter = 2;
  Eigen::LLT<MatrixXd> llt;

  while (std::abs(dev1 - dev2) > tol && iter < maxiter) {
    iter++;
    dev1 = dev2;

    w.array() = mu.array().max(1e-8);
    z.array() = eta.array() + (y.array() - mu.array()) / w.array();

    XtWX.noalias() = Xt * w.asDiagonal() * X_des;
    XtWz.noalias() = Xt * (w.array() * z.array()).matrix();

    llt.compute(XtWX);
    beta = llt.solve(XtWz);

    eta.noalias() = X_des * beta;

    for (int i = 0; i < n; ++i) {
      mu(i) = std::exp(eta(i));
    }

    dev2 = 0.0;
    for (int i = 0; i < n; ++i) {
      if (y(i) > 0) {
        dev2 += y(i) * std::log(y(i) / mu(i));
      }
      dev2 += mu(i) - y(i);
    }
    dev2 *= 2.0;
  }

  MatrixXd vcov = llt.solve(MatrixXd::Identity(p, p));
  se = vcov.diagonal().array().sqrt();

  return Rcpp::List::create(Named("coefficients") = beta, Named("vcov") = vcov,
                            Named("se") = se, Named("dev") = dev2,
                            Named("iters") = iter);
}
