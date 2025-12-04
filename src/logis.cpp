// [[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

Eigen::MatrixXd logis_parallel(const Eigen::MatrixXd &X,
                               const Eigen::VectorXd &y, double tol = 1e-6,
                               int maxiter = 100) {
  const int n = X.rows();
  const int p = X.cols();

  Eigen::MatrixXd res(p, 3);

  Eigen::VectorXd eta(n), mu(n);
  const double sum_w_init = 0.25 * n;

  Eigen::Array<bool, Eigen::Dynamic, 1> y_is_one(n);
  for (int i = 0; i < n; ++i) {
    y_is_one(i) = (y(i) > 0.5);
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < p; ++j) {
    const auto x = X.col(j);

    double sum_wx = 0.0;
    double sum_wx2 = 0.0;
    double sum_wz = 0.0;
    double sum_wxz = 0.0;

    for (int i = 0; i < n; ++i) {
      const double x_i = x(i);
      const double z_i = y_is_one(i) ? 2.0 : -2.0;

      sum_wx += x_i;
      sum_wx2 += x_i * x_i;
      sum_wz += z_i;
      sum_wxz += x_i * z_i;
    }

    sum_wx *= 0.25;
    sum_wx2 *= 0.25;
    sum_wz *= 0.25;
    sum_wxz *= 0.25;

    double det = sum_w_init * sum_wx2 - sum_wx * sum_wx;
    double beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) / det;
    double beta1 = (sum_w_init * sum_wxz - sum_wx * sum_wz) / det;

    double dev_old = 0.0;
    for (int i = 0; i < n; ++i) {
      eta(i) = beta0 + beta1 * x(i);
      const double exp_neg_eta = std::exp(-eta(i));
      mu(i) = 1.0 / (1.0 + exp_neg_eta);

      dev_old -=
          y_is_one(i) ? std::log(mu(i) + 1e-16) : std::log(1.0 - mu(i) + 1e-16);
    }

    for (int iter = 1; iter < maxiter; ++iter) {
      double sum_w = 0.0;
      sum_wx = sum_wx2 = sum_wz = sum_wxz = 0.0;

      for (int i = 0; i < n; ++i) {
        const double mu_i = mu(i);
        const double w_i = std::max(mu_i * (1.0 - mu_i), 1e-8);
        const double x_i = x(i);
        const double z_i = eta(i) + (y(i) - mu_i) / w_i;

        sum_w += w_i;
        const double w_x = w_i * x_i;
        sum_wx += w_x;
        sum_wx2 += w_x * x_i;
        sum_wz += w_i * z_i;
        sum_wxz += w_x * z_i;
      }

      det = sum_w * sum_wx2 - sum_wx * sum_wx;
      beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) / det;
      beta1 = (sum_w * sum_wxz - sum_wx * sum_wz) / det;

      double dev_new = 0.0;
      for (int i = 0; i < n; ++i) {
        eta(i) = beta0 + beta1 * x(i);
        const double exp_neg_eta = std::exp(-eta(i));
        mu(i) = 1.0 / (1.0 + exp_neg_eta);

        dev_new -= y_is_one(i) ? std::log(mu(i) + 1e-16)
                               : std::log(1.0 - mu(i) + 1e-16);
      }

      if (std::abs(dev_old - dev_new) <= tol)
        break;

      dev_old = dev_new;
    }

    res(j, 0) = beta0;
    res(j, 1) = beta1;
    res(j, 2) = 2.0 * dev_old;
  }

  return res;
}

Eigen::MatrixXd logis_cpp(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                          double tol = 1e-6, int maxiter = 100) {
  const int n = X.rows();
  const int p = X.cols();
  Eigen::MatrixXd res(p, 3);
  Eigen::VectorXd eta(n), mu(n);
  const double sum_w_init = 0.25 * n;

  Eigen::Array<bool, Eigen::Dynamic, 1> y_is_one(n);
  for (int i = 0; i < n; ++i) {
    y_is_one(i) = (y(i) > 0.5);
  }

  for (int j = 0; j < p; ++j) {
    const auto x = X.col(j);

    double sum_wx = 0.0;
    double sum_wx2 = 0.0;
    double sum_wz = 0.0;
    double sum_wxz = 0.0;

    for (int i = 0; i < n; ++i) {
      const double x_i = x(i);
      const double z_i = y_is_one(i) ? 2.0 : -2.0;
      sum_wx += x_i;
      sum_wx2 += x_i * x_i;
      sum_wz += z_i;
      sum_wxz += x_i * z_i;
    }

    sum_wx *= 0.25;
    sum_wx2 *= 0.25;
    sum_wz *= 0.25;
    sum_wxz *= 0.25;

    double det = sum_w_init * sum_wx2 - sum_wx * sum_wx;
    double beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) / det;
    double beta1 = (sum_w_init * sum_wxz - sum_wx * sum_wz) / det;

    double dev_old = 0.0;
    for (int i = 0; i < n; ++i) {
      eta(i) = beta0 + beta1 * x(i);
      const double exp_neg_eta = std::exp(-eta(i));
      mu(i) = 1.0 / (1.0 + exp_neg_eta);

      if (y_is_one(i)) {
        dev_old -= std::log(mu(i) + 1e-16);
      } else {
        dev_old -= std::log(1.0 - mu(i) + 1e-16);
      }
    }

    for (int iter = 1; iter < maxiter; ++iter) {
      double sum_w = 0.0;
      sum_wx = 0.0;
      sum_wx2 = 0.0;
      double sum_wz = 0.0;
      double sum_wxz = 0.0;

      for (int i = 0; i < n; ++i) {
        const double mu_i = mu(i);
        const double w_i = std::max(mu_i * (1.0 - mu_i), 1e-8);
        const double x_i = x(i);
        const double z_i = eta(i) + (y(i) - mu_i) / w_i;

        sum_w += w_i;
        const double w_x = w_i * x_i;
        sum_wx += w_x;
        sum_wx2 += w_x * x_i;
        sum_wz += w_i * z_i;
        sum_wxz += w_x * z_i;
      }

      det = sum_w * sum_wx2 - sum_wx * sum_wx;
      beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) / det;
      beta1 = (sum_w * sum_wxz - sum_wx * sum_wz) / det;

      double dev_new = 0.0;
      for (int i = 0; i < n; ++i) {
        eta(i) = beta0 + beta1 * x(i);
        const double exp_neg_eta = std::exp(-eta(i));
        mu(i) = 1.0 / (1.0 + exp_neg_eta);

        if (y_is_one(i)) {
          dev_new -= std::log(mu(i) + 1e-16);
        } else {
          dev_new -= std::log(1.0 - mu(i) + 1e-16);
        }
      }

      if (std::abs(dev_old - dev_new) <= tol) {
        break;
      }
      dev_old = dev_new;
    }

    res(j, 0) = beta0;
    res(j, 1) = beta1;
    res(j, 2) = 2.0 * dev_old;
  }

  return res;
}
//[[Rcpp::export]]

Eigen::MatrixXd logis_new(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                          double tol = 1e-6, int maxiter = 100,
                          bool parallel = false) {
  if (parallel)
    return logis_parallel(X, y, tol, maxiter);
  return logis_cpp(X, y, tol, maxiter);
}
