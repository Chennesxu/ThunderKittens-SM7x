#pragma once

#include <cmath>
#include <limits>

namespace tk_sm7x::test::numerical {

inline double error_bound(int k, double sum_abs) {
  if (k <= 0 || k % 16 != 0 || k > 1024 || sum_abs < 0.0 ||
      !std::isfinite(sum_abs)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (sum_abs == 0.0) {
    return 0.0;
  }

  constexpr double epsilon = 0x1p-23;
  const double n = 2.0 * static_cast<double>(k);
  const double gamma_up = std::nextafter(
      n * epsilon / (1.0 - n * epsilon), std::numeric_limits<double>::infinity());
  const double bound = std::nextafter(
      gamma_up * sum_abs, std::numeric_limits<double>::infinity());
  if (!std::isfinite(bound)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return bound;
}

inline bool within_bound(double actual, double reference, double bound) {
  return std::isfinite(actual) && std::isfinite(reference) && std::isfinite(bound) &&
         bound >= 0.0 && std::fabs(actual - reference) <= bound;
}

}  // namespace tk_sm7x::test::numerical
