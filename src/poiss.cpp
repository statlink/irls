// [[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

Eigen::MatrixXd poiss_cpp(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                          double tol = 1e-6, int maxiter = 100) {

  const int n = X.rows();
  const int p = X.cols();

  Eigen::MatrixXd res(p, 3);
  Eigen::VectorXd eta(n), mu(n);

  const double w_min = 1e-8;
  const double eps = 1e-16;

  const double mean_y = y.sum() / n;
  const double eta_init = std::log(mean_y);
  const double mu_init = mean_y;
  const double w_init = mu_init;
  const double sum_w_init = w_init * n;
  const double inv_w_init = 1.0 / w_init;

  Eigen::VectorXd y_log_y(n);
  for (int i = 0; i < n; ++i) {
    y_log_y(i) = (y(i) > eps) ? y(i) * std::log(y(i)) : 0.0;
  }

  for (int j = 0; j < p; ++j) {

    const double *x_ptr = X.data() + j * n;

    double sum_wx = 0.0;
    double sum_wx2 = 0.0;
    double sum_wz = 0.0;
    double sum_wxz = 0.0;

    for (int i = 0; i < n; ++i) {
      const double x_i = x_ptr[i];
      const double z_i = eta_init + (y(i) - mu_init) * inv_w_init;
      const double wx = w_init * x_i;

      sum_wx += wx;
      sum_wx2 += wx * x_i;
      sum_wz += w_init * z_i;
      sum_wxz += wx * z_i;
    }

    double det_inv = 1.0 / (sum_w_init * sum_wx2 - sum_wx * sum_wx);
    double beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) * det_inv;
    double beta1 = (sum_w_init * sum_wxz - sum_wx * sum_wz) * det_inv;

    double dev_old = 0.0;
    for (int i = 0; i < n; ++i) {
      const double eta_i = beta0 + beta1 * x_ptr[i];
      eta(i) = eta_i;
      const double mu_i = std::exp(eta_i);
      mu(i) = mu_i;

      if (y(i) > eps) {
        dev_old += y_log_y(i) - y(i) * std::log(mu_i);
      }
      dev_old -= (y(i) - mu_i);
    }

    for (int iter = 1; iter < maxiter; ++iter) {

      double sum_w = 0.0;
      sum_wx = 0.0;
      sum_wx2 = 0.0;
      sum_wz = 0.0;
      sum_wxz = 0.0;

      for (int i = 0; i < n; ++i) {
        const double mu_i = mu(i);
        const double w_i = std::max(mu_i, w_min);
        const double inv_w = 1.0 / w_i;
        const double z_i = eta(i) + (y(i) - mu_i) * inv_w;
        const double x_i = x_ptr[i];
        const double wx = w_i * x_i;

        sum_w += w_i;
        sum_wx += wx;
        sum_wx2 += wx * x_i;
        sum_wz += w_i * z_i;
        sum_wxz += wx * z_i;
      }

      det_inv = 1.0 / (sum_w * sum_wx2 - sum_wx * sum_wx);
      beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) * det_inv;
      beta1 = (sum_w * sum_wxz - sum_wx * sum_wz) * det_inv;

      double dev_new = 0.0;
      for (int i = 0; i < n; ++i) {
        const double eta_i = beta0 + beta1 * x_ptr[i];
        eta(i) = eta_i;
        const double mu_i = std::exp(eta_i);
        mu(i) = mu_i;

        if (y(i) > eps) {
          dev_new += y_log_y(i) - y(i) * std::log(mu_i);
        }
        dev_new -= (y(i) - mu_i);
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

Eigen::MatrixXd poiss_parallel(const Eigen::MatrixXd &X,
                               const Eigen::VectorXd &y, double tol = 1e-6,
                               int maxiter = 100) {

  const int n = X.rows();
  const int p = X.cols();

  Eigen::MatrixXd res(p, 3);
  Eigen::VectorXd eta(n), mu(n);

  const double w_min = 1e-8;
  const double eps = 1e-16;

  const double mean_y = y.sum() / n;
  const double eta_init = std::log(mean_y);
  const double mu_init = mean_y;
  const double w_init = mu_init;
  const double sum_w_init = w_init * n;
  const double inv_w_init = 1.0 / w_init;

  Eigen::VectorXd y_log_y(n);
  for (int i = 0; i < n; ++i) {
    y_log_y(i) = (y(i) > eps) ? y(i) * std::log(y(i)) : 0.0;
  }
#ifdef _OPENMP
#pragma omp parallel for
#endif

  for (int j = 0; j < p; ++j) {

    const double *x_ptr = X.data() + j * n;

    double sum_wx = 0.0;
    double sum_wx2 = 0.0;
    double sum_wz = 0.0;
    double sum_wxz = 0.0;

    for (int i = 0; i < n; ++i) {
      const double x_i = x_ptr[i];
      const double z_i = eta_init + (y(i) - mu_init) * inv_w_init;
      const double wx = w_init * x_i;

      sum_wx += wx;
      sum_wx2 += wx * x_i;
      sum_wz += w_init * z_i;
      sum_wxz += wx * z_i;
    }

    double det_inv = 1.0 / (sum_w_init * sum_wx2 - sum_wx * sum_wx);
    double beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) * det_inv;
    double beta1 = (sum_w_init * sum_wxz - sum_wx * sum_wz) * det_inv;

    double dev_old = 0.0;
    for (int i = 0; i < n; ++i) {
      const double eta_i = beta0 + beta1 * x_ptr[i];
      eta(i) = eta_i;
      const double mu_i = std::exp(eta_i);
      mu(i) = mu_i;

      if (y(i) > eps) {
        dev_old += y_log_y(i) - y(i) * std::log(mu_i);
      }
      dev_old -= (y(i) - mu_i);
    }

    for (int iter = 1; iter < maxiter; ++iter) {

      double sum_w = 0.0;
      sum_wx = 0.0;
      sum_wx2 = 0.0;
      sum_wz = 0.0;
      sum_wxz = 0.0;

      for (int i = 0; i < n; ++i) {
        const double mu_i = mu(i);
        const double w_i = std::max(mu_i, w_min);
        const double inv_w = 1.0 / w_i;
        const double z_i = eta(i) + (y(i) - mu_i) * inv_w;
        const double x_i = x_ptr[i];
        const double wx = w_i * x_i;

        sum_w += w_i;
        sum_wx += wx;
        sum_wx2 += wx * x_i;
        sum_wz += w_i * z_i;
        sum_wxz += wx * z_i;
      }

      det_inv = 1.0 / (sum_w * sum_wx2 - sum_wx * sum_wx);
      beta0 = (sum_wx2 * sum_wz - sum_wx * sum_wxz) * det_inv;
      beta1 = (sum_w * sum_wxz - sum_wx * sum_wz) * det_inv;

      double dev_new = 0.0;
      for (int i = 0; i < n; ++i) {
        const double eta_i = beta0 + beta1 * x_ptr[i];
        eta(i) = eta_i;
        const double mu_i = std::exp(eta_i);
        mu(i) = mu_i;

        if (y(i) > eps) {
          dev_new += y_log_y(i) - y(i) * std::log(mu_i);
        }
        dev_new -= (y(i) - mu_i);
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

Eigen::MatrixXd poiss_new(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                      double tol = 1e-6, int maxiter = 100,
                      bool parallel = false) {
  if (parallel)
    return poiss_parallel(X, y, tol, maxiter);
  return poiss_cpp(X, y, tol, maxiter);
}
