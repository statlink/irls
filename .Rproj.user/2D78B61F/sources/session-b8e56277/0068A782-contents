// [[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif


Eigen::MatrixXd gammas_cpp(const Eigen::MatrixXd &X, 
                                     const Eigen::VectorXd &y,
                                     double tol = 1e-8, int maxiter = 100) {
    const int n = X.rows();
    const int p = X.cols();
    Eigen::MatrixXd res(p, 4);
    
    const Eigen::VectorXd log_y = y.array().log();
    const double df_residual = n - 2.0;
    
    std::vector<double> xx00(p), xx01(p), xx11(p);
    
    for (int j = 0; j < p; ++j) {
        const double* x_ptr = X.data() + j * n;
        double sx = 0.0, sx2 = 0.0;
        
        for (int i = 0; i < n; ++i) {
            const double x_i = x_ptr[i];
            sx  += x_i;
            sx2 += x_i * x_i;
        }
        
        const double det_inv = 1.0 / (n * sx2 - sx * sx);
        xx00[j] =  sx2 * det_inv;
        xx01[j] = -sx  * det_inv;
        xx11[j] =  n   * det_inv;
    }
    
    Eigen::VectorXd eta(n), mu(n);
    
    for (int j = 0; j < p; ++j) {
        const double* x_ptr = X.data() + j * n;
        double* eta_ptr = eta.data();
        double* mu_ptr  = mu.data();
        const double* log_y_ptr = log_y.data();
        const double* y_ptr = y.data();
        
        double sum_log_y = 0.0;
        double sum_x_log_y = 0.0;
        
        for (int i = 0; i < n; ++i) {
            const double log_y_i = log_y_ptr[i];
            sum_log_y   += log_y_i;
            sum_x_log_y += x_ptr[i] * log_y_i;
        }
        
        double beta0 = xx00[j] * sum_log_y + xx01[j] * sum_x_log_y;
        double beta1 = xx01[j] * sum_log_y + xx11[j] * sum_x_log_y;
        
        double dev_old = 0.0;
        for (int i = 0; i < n; ++i) {
            const double eta_i = beta0 + beta1 * x_ptr[i];
            eta_ptr[i] = eta_i;
            const double mu_i = std::exp(eta_i);
            mu_ptr[i] = mu_i;
            
            const double y_i = y_ptr[i];
            const double y_over_mu = y_i / mu_i;
            dev_old += 2.0 * (y_over_mu - 1.0 - log_y_ptr[i] + eta_i);
        }
        
        for (int iter = 1; iter <= maxiter; ++iter) {
            double sum_z = 0.0;
            double sum_xz = 0.0;
            
            for (int i = 0; i < n; ++i) {
                const double mu_i = mu_ptr[i];
                const double inv_mu = 1.0 / mu_i;
                const double z_i = eta_ptr[i] + (y_ptr[i] - mu_i) * inv_mu;
                
                sum_z  += z_i;
                sum_xz += x_ptr[i] * z_i;
            }
            
            beta0 = xx00[j] * sum_z + xx01[j] * sum_xz;
            beta1 = xx01[j] * sum_z + xx11[j] * sum_xz;
            
            double dev_new = 0.0;
            for (int i = 0; i < n; ++i) {
                const double eta_i = beta0 + beta1 * x_ptr[i];
                eta_ptr[i] = eta_i;
                const double mu_i = std::exp(eta_i);
                mu_ptr[i] = mu_i;
                
                const double y_i = y_ptr[i];
                const double y_over_mu = y_i / mu_i;
                dev_new += 2.0 * (y_over_mu - 1.0 - log_y_ptr[i] + eta_i);
            }
            
            const double rel_change = std::abs(dev_new - dev_old) / (0.1 + std::abs(dev_new));
            if (rel_change < tol) {
                dev_old = dev_new;
                break;
            }
            
            dev_old = dev_new;
        }
        
        double pearson = 0.0;
        for (int i = 0; i < n; ++i) {
            const double inv_mu = 1.0 / mu_ptr[i];
            const double resid = (y_ptr[i] - mu_ptr[i]) * inv_mu;
            pearson += resid * resid;
        }
        
        const double phi = pearson / df_residual;
        
        res(j, 0) = beta0;
        res(j, 1) = beta1;
        res(j, 2) = dev_old;
        res(j, 3) = phi;
    }
    
    return res;
}
Eigen::MatrixXd gammas_cpp_parallel(const Eigen::MatrixXd &X,
                                    const Eigen::VectorXd &y, 
                                    double tol = 1e-8,
                                    int maxiter = 100) {
  const int n = X.rows();
  const int p = X.cols();
  
  Eigen::MatrixXd res(p, 4);
  
  Eigen::VectorXd log_y(n);
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; ++i) {
    log_y(i) = std::log(y(i));
  }
  
  const double df_residual = n - 2.0;
  
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Eigen::VectorXd eta(n), mu(n);
    
#ifdef _OPENMP
#pragma omp for
#endif
    for (int j = 0; j < p; ++j) {
      
      const double* x_ptr = X.data() + j * n;
      
      double sum_x = 0.0;
      double sum_x2 = 0.0;
      
      for (int i = 0; i < n; ++i) {
        const double x_i = x_ptr[i];
        sum_x += x_i;
        sum_x2 += x_i * x_i;
      }
      
      const double det_inv = 1.0 / (n * sum_x2 - sum_x * sum_x);
      const double xx_inv_00 = sum_x2 * det_inv;
      const double xx_inv_01 = -sum_x * det_inv;
      const double xx_inv_11 = n * det_inv;
      
      double sum_log_y = 0.0;
      double sum_x_log_y = 0.0;
      
      for (int i = 0; i < n; ++i) {
        const double log_y_i = log_y(i);
        sum_log_y += log_y_i;
        sum_x_log_y += x_ptr[i] * log_y_i;
      }
      
      double beta0 = xx_inv_00 * sum_log_y + xx_inv_01 * sum_x_log_y;
      double beta1 = xx_inv_01 * sum_log_y + xx_inv_11 * sum_x_log_y;
      
      double dev_old = 0.0;
      for (int i = 0; i < n; ++i) {
        const double eta_i = beta0 + beta1 * x_ptr[i];
        eta(i) = eta_i;
        const double mu_i = std::exp(eta_i);
        mu(i) = mu_i;
        
        const double y_i = y(i);
        const double inv_mu = 1.0 / mu_i;
        const double ratio = y_i * inv_mu;
        dev_old += 2.0 * ((y_i - mu_i) * inv_mu - std::log(ratio));
      }
      
      for (int iter = 1; iter <= maxiter; ++iter) {
        
        double sum_z = 0.0;
        double sum_xz = 0.0;
        
        for (int i = 0; i < n; ++i) {
          const double mu_i = mu(i);
          const double inv_mu = 1.0 / mu_i;
          const double z_i = eta(i) + (y(i) - mu_i) * inv_mu;
          
          sum_z += z_i;
          sum_xz += x_ptr[i] * z_i;
        }
        
        beta0 = xx_inv_00 * sum_z + xx_inv_01 * sum_xz;
        beta1 = xx_inv_01 * sum_z + xx_inv_11 * sum_xz;
        
        double dev_new = 0.0;
        for (int i = 0; i < n; ++i) {
          const double eta_i = beta0 + beta1 * x_ptr[i];
          eta(i) = eta_i;
          const double mu_i = std::exp(eta_i);
          mu(i) = mu_i;
          
          const double y_i = y(i);
          const double inv_mu = 1.0 / mu_i;
          const double ratio = y_i * inv_mu;
          dev_new += 2.0 * ((y_i - mu_i) * inv_mu - std::log(ratio));
        }
        
        const double rel_change = std::abs(dev_new - dev_old) / (0.1 + std::abs(dev_new));
        if (rel_change < tol) {
          dev_old = dev_new;
          break;
        }
        
        dev_old = dev_new;
      }
      
      double pearson = 0.0;
      for (int i = 0; i < n; ++i) {
        const double inv_mu = 1.0 / mu(i);
        const double resid = (y(i) - mu(i)) * inv_mu;
        pearson += resid * resid;
      }
      
      const double phi = pearson / df_residual;
      
      res(j, 0) = beta0;
      res(j, 1) = beta1;
      res(j, 2) = dev_old;
      res(j, 3) = phi;
    }
  } 
  
  return res;
}
//[[Rcpp::export]]

Eigen::MatrixXd gammas_new(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                           double tol = 1e-8, int maxiter = 100,
                           bool parallel = false) {
  if (parallel)
    return gammas_cpp_parallel(X, y, tol, maxiter);
  return gammas_cpp(X, y, tol, maxiter);
}
