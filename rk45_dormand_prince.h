#pragma once

#include <vector>
#include <cmath>
#include <iostream>

class rk45_dormand_prince {
public:
  rk45_dormand_prince(int num_equations, double tolerance_abs, double tolerance_rel) {
    // constructor, initialize the data members
    n_eq = num_equations;
    atol = tolerance_abs;
    rtol = tolerance_rel;
    y_tmp.resize(n_eq);
    y_err.resize(n_eq);

    k1.resize(n_eq);
    k2.resize(n_eq);
    k3.resize(n_eq);
    k4.resize(n_eq);
    k5.resize(n_eq);
    k6.resize(n_eq);
    k7.resize(n_eq);

    r1.resize(n_eq);
    r2.resize(n_eq);
    r3.resize(n_eq);
    r4.resize(n_eq);
    r5.resize(n_eq);
  }

  template <typename F, typename StopCond>
  std::vector<double> integrate(const F &f, const StopCond &stop, double x0, 
                                const std::vector<double> &y0,
                                bool dense_output = false,
                                const std::vector<double> &x_out = {}) {
    // clear the arrays so that we start fresh every time we call this function
    xs.clear();
    result.clear();
    hs.clear();

    // initial conditions
    double h = 0.1; // arbitrary initial step size
    double x = x0;
    std::vector<double> y = y0;
    double err = 1.0;
    // err_prev is for you to implement the more advanced step size control
    double err_prev = 1.0;
    k1 = f(x, y);

    if (!dense_output) {
      // If we are not using dense output, we need to store the initial condition
      hs.push_back(h);
      xs.push_back(x);
      result.push_back(y);
    }

    std::vector<double> y_prev = y;
    std::vector<double> y_next = y;
    std::vector<double> y_out(n_eq); // This stores the value for dense output
    int xout_idx = 0; // This is the index for the x_out array

    while (!stop(x, y_prev, y)) {
      y_next = step(f, h, x, y);
      err = error(y_next);
      // If err is fortuitously too small, set it to some lower bound
      err = std::max(err, 1.0e-10);

      // Accept the step if the scalar error is below 1, otherwise reject it and
      // do not move forward
      if (err < 1.0) {
        if (dense_output) {
          // Compute dense output coefficients
          compute_dense_rs(y_next, y, h);

          // Loop through the output array and interpolate the values
          while (x_out[xout_idx] < x + h && xout_idx < x_out.size()) {
            double x_out_val = x_out[xout_idx];
            double th = (x_out_val - x) / h;
            if (th < 0.0) { // If the initial x_out value is less than x, set th to 0
              th = 0.0;
            }
            for (int i = 0; i < n_eq; i++) {
              y_out[i] = r1[i] + th * (r2[i] + (1.0 - th) * (r3[i] + th * (r4[i] + (1.0 - th) * r5[i])));
            }
            xs.push_back(x_out_val);
            result.push_back(y_out);
            xout_idx++;
          }

          x += h;
        } else {
          x += h;
          // This is what we do when not using dense output
          hs.push_back(h);
          xs.push_back(x);
          result.push_back(y_next);

        }
        k1 = k7;
        // store the current error as err_prev
        err_prev = err;
        y_prev = y;
        y = y_next;
      }

      // Adjust h as needed
      double S = 0.9; // safety factor
      if (err_prev < 1.0) {
        // If the previous step was accepted, we can use the step size control
        double err_alpha = 0.7 / 5.0;
        double err_beta = 0.4 / 5.0;
        h = S * h * std::pow(err, -err_alpha) * std::pow(err_prev, err_beta);
      } else {
        // If the previous step was rejected, do not increase the step size
        h = std::min(h, S * h * std::pow(1.0 / err, 0.2));
      }
      h = std::max(h, hmin);
      h = std::min(h, hmax);

      // Uncomment the following line to see the step size and error at each step
      // std::cout << "x = " << x << ", h = " << h << ", err = " << err << std::endl;
    }
    return y_next;
  }

  template <typename F>
  std::vector<double> step(const F& f, double h, double x, const std::vector<double> &y) {
    // Compute the next step in y, given x and y of the current step
    std::vector<double> y_next(n_eq);

    // First step
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * a21 * k1[i];
    }

    // Second step
    k2 = f(x + c2 * h, y_tmp);
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    }

    // Third step
    k3 = f(x + c3 * h, y_tmp);
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    }

    // Fourth step
    k4 = f(x + c4 * h, y_tmp);
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] +
                              a54 * k4[i]);
    }

    // Fifth step
    k5 = f(x + c5 * h, y_tmp);
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] +
                              a64 * k4[i] + a65 * k5[i]);
    }

    // Sixth step
    k6 = f(x + h, y_tmp);
    for (int i = 0; i < n_eq; i++) {
      y_next[i] = y[i] + h * (a71 * k1[i] + a72 * k2[i] + a73 * k3[i] +
                              a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
    }
    k7 = f(x + h, y_next);

    // Estimate y_err for each y in the vector using the difference
    // between y1 and y2
    for (int i = 0; i < n_eq; i++) {
      y_err[i] = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] +
                       e6 * k6[i] + e7 * k7[i]);
    }

    return y_next;
  }

  double error(const std::vector<double> &y) {
    double err = 0.0;
    for (int i = 0; i < n_eq; i++) {
      double scale =
          atol + std::max(std::abs(y[i]), std::abs(y[i])) * rtol;
      err += std::pow(y_err[i] / scale, 2);
    }
    return std::sqrt(err / n_eq);
  }

  void compute_dense_rs(const std::vector<double> &y_next, const std::vector<double> &y,
                        double h) {
    // Compute the coefficients r1, r2, r3, r4, r5
    for (int i = 0; i < n_eq; i++) {
      r1[i] = y[i];
      r2[i] = y_next[i] - y[i];
      r3[i] = y[i] + h * k1[i] - y_next[i];
      r4[i] = 2.0 * (y_next[i] - y[i]) - h * (k1[i] + k7[i]);
      r5[i] = h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] +
                   d6 * k6[i] + d7 * k7[i]);
    }
  }

  int n_eq;
  double atol, rtol;

  // We impose a minimum and maximum step size
  const double hmin = 1.0e-10;
  const double hmax = 1.0;

  // These are temporary variables used to store the coefficients. You are allowed to
  // define additional temporary variables if you need them.
  std::vector<double> k1, k2, k3, k4, k5, k6, k7, y_tmp, y_err;

  // vectors that store the results
  std::vector<double> hs;
  std::vector<double> xs;
  std::vector<std::vector<double>> result;

  // These are temporary variables used to store the coefficients
  // used in dense output
  std::vector<double> r1, r2, r3, r4, r5;

  // c1 is zero, c6 and c7 are 1.0
  const double c2 = 1.0 / 5.0;
  const double c3 = 3.0 / 10.0;
  const double c4 = 4.0 / 5.0;
  const double c5 = 8.0 / 9.0;

  const double a21 = 1.0 / 5.0;
  const double a31 = 3.0 / 40.0;
  const double a32 = 9.0 / 40.0;
  const double a41 = 44.0 / 45.0;
  const double a42 = -56.0 / 15.0;
  const double a43 = 32.0 / 9.0;
  const double a51 = 19372.0 / 6561.0;
  const double a52 = -25360.0 / 2187.0;
  const double a53 = 64448.0 / 6561.0;
  const double a54 = -212.0 / 729.0;
  const double a61 = 9017.0 / 3168.0;
  const double a62 = -355.0 / 33.0;
  const double a63 = 46732.0 / 5247.0;
  const double a64 = 49.0 / 176.0;
  const double a65 = -5103.0 / 18656.0;

  // Note that a71, a72, a73, a74, a75, a76 are essentially the coefficients of the 5th order solution
  const double a71 = 35.0 / 384.0;
  const double a72 = 0.0;
  const double a73 = 500.0 / 1113.0;
  const double a74 = 125.0 / 192.0;
  const double a75 = -2187.0 / 6784.0;
  const double a76 = 11.0 / 84.0;

  // These coefficients are used to estimate the error in the solution. They are essentially 
  // the coefficients of the 5th order solution minus the 4th order solution, i.e. b_i - b_i^*.
  const double e1 = 71.0 / 57600.0;
  const double e2 = 0.0;
  const double e3 = -71.0 / 16695.0;
  const double e4 = 71.0 / 1920.0;
  const double e5 = -17253.0 / 339200.0;
  const double e6 = 22.0 / 525.0;
  const double e7 = -1.0 / 40.0;

  // These are the coefficients for dense output
  const double d1 = -12715105075.0 / 11282082432.0;
  const double d2 = 0.0;
  const double d3 = 87487479700.0 / 32700410799.0;
  const double d4 = -10690763975.0 / 1880347072.0;
  const double d5 = 701980252875.0 / 199316789632.0;
  const double d6 = -1453857185.0 / 822651844.0;
  const double d7 = 69997945.0 / 29380423.0;
};
