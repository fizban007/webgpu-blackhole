#pragma once

#include <cmath>

double square(double x) { return x * x; }

double cube(double x) { return x * x * x; }

class boyer_lindquist_metric {
public:
  boyer_lindquist_metric(double a0, double M0 = 1.0) : a(a0), M(M0) {}
  ~boyer_lindquist_metric() {}

  void compute_metric(double r, double th) {
    double sth = std::sin(th);
    double cth = std::cos(th);
    double c2th = std::cos(2.0 * th);
    double s2th = std::sin(2.0 * th);
    double cscth = 1.0 / sth;
    double a2 = a * a;
    double a3 = a * a2;
    double a4 = a2 * a2;
    double a5 = a4 * a;
    double a6 = a2 * a4;
    double r2 = r * r;
    double r3 = r2 * r;
    double r4 = r2 * r2;
    double r6 = r2 * r4;

    delta = r * r - 2.0 * M * r + a * a;
    sigma = square(r2 + a2) - a2 * delta * square(sth);
    rho2 = r2 + a * a * cth * cth;

    alpha = std::sqrt(rho2 * delta / sigma);
    beta3 = -2.0 * M * a * r / sigma;

    g_00 = 2.0 * M * r / rho2 - 1.0;
    g_03 = -2.0 * M * a * r / rho2 * square(sth);
    g_11 = rho2 / delta;
    g_22 = rho2;
    g_33 = sigma * square(sth) / rho2;

    gamma11 = delta / rho2;
    gamma22 = 1.0 / rho2;
    gamma33 = rho2 / sigma / square(sth);

    d_alpha_dr = M *
                 (-a6 + 2.0 * r6 + a2 * r3 * (3.0 * r - 4.0 * M) -
                  a2 * (a4 + 2.0 * a2 * r2 + r3 * (r - 4.0 * M)) * c2th) /
                 (2.0 * sigma * sigma * std::sqrt(delta * rho2 / sigma));
    d_beta3_dr = M *
                 (-a5 + 3.0 * a3 * r2 + 6.0 * a * r4 + a3 * (r2 - a2) * c2th) /
                 square(sigma);
    d_gamma11_dr = 2.0 * (r * (M * r - a2) + a2 * (r - 1.0 * M) * square(cth)) /
                   square(rho2);
    d_gamma22_dr = -2.0 * r / square(rho2);
    d_gamma33_dr =
        (-2.0 * a4 * (r - 1.0 * M) * square(cscth) +
         2.0 * (a2 * (2.0 * r - 1.0 * M) + r2 * (2.0 * r + 1.0 * M)) *
             square(a * square(cscth)) -
         2.0 * r * square(a2 + r2) * square(cube(cscth))) /
        square(a4 + a2 * r * (r - 2.0 * M) - square((a2 + r2) * cscth));
    d_alpha_dth = -M * a2 * delta * r * (a2 + r2) * s2th / square(sigma) /
                  std::sqrt(delta * rho2 / sigma);
    d_beta3_dth = -2.0 * M * a3 * r * delta * s2th / square(sigma);
    d_gamma11_dth = a2 * delta * s2th / square(rho2);
    d_gamma22_dth = a2 * s2th / square(rho2);
    d_gamma33_dth =
        2.0 *
        (-a4 * delta + 2.0 * a2 * delta * (a2 + r2) * square(cscth) -
         cube(a2 + r2) * square(square(cscth))) *
        cth / cube(sth) /
        square(a4 + a2 * r * (r - 2.0 * M) - square((a2 + r2) * cscth));

  }

  double u0(double u_1, double u_2, double u_3) {
    return std::sqrt(gamma11 * u_1 * u_1 + gamma22 * u_2 * u_2 +
                     gamma33 * u_3 * u_3) /
           alpha;
  }

  double u_0(double u_1, double u_2, double u_3) {
    return -alpha * alpha * u0(u_1, u_2, u_3) + beta3 * u_3;
  }

  double a = 0.0;
  double M = 1.0;
  double alpha, beta3;
  double gamma11, gamma22, gamma33;
  double g_00, g_11, g_22, g_33, g_03;
  double d_alpha_dr, d_beta3_dr, d_gamma11_dr, d_gamma22_dr, d_gamma33_dr;
  double d_alpha_dth, d_beta3_dth, d_gamma11_dth, d_gamma22_dth, d_gamma33_dth;

  double g00, g11, g22, g33, g03;
  double d_g00_dr, d_g03_dr, d_g11_dr, d_g22_dr, d_g33_dr;
  double d_g00_dth, d_g03_dth, d_g11_dth, d_g22_dth, d_g33_dth;
  double delta, sigma, rho2;
};
