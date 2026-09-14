#include "numerical_bounds.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace {

class checks {
 public:
  void expect(bool condition, const char* description) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << description << '\n';
    }
  }

  int result() const {
    if (failures_ == 0) {
      std::cout << "numerical bounds: PASS\n";
      return 0;
    }
    std::cerr << "numerical bounds: " << failures_ << " failure(s)\n";
    return 1;
  }

 private:
  int failures_ = 0;
};

}  // namespace

int main() {
  using tk_sm7x::test::numerical::error_bound;
  using tk_sm7x::test::numerical::within_bound;

  checks test;
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const long double expected_gamma = 1.0L / 262143.0L;
  const double gamma = error_bound(16, 1.0);
  test.expect(std::isfinite(gamma) && static_cast<long double>(gamma) >= expected_gamma,
              "K=16, S=1 rounds the analytical gamma budget outward");
  test.expect(error_bound(16, 262143.0) >= 1.0 &&
                  error_bound(16, 262143.0) <=
                      std::nextafter(std::nextafter(std::nextafter(std::nextafter(
                          1.0, infinity), infinity), infinity), infinity),
              "K=16, S=262143 is only a few double ULPs above one");
  test.expect(error_bound(1024, 1.0) > 0.0 && std::isfinite(error_bound(1024, 1.0)),
              "largest valid K has a finite positive budget");
  test.expect(error_bound(16, 0.0) == 0.0 && error_bound(16, -0.0) == 0.0,
              "positive and signed zero scales have a zero budget");
  test.expect(!std::signbit(error_bound(16, -0.0)),
              "signed zero scale produces positive zero budget");

  test.expect(within_bound(0.25, 0.0, 0.25),
              "inclusive comparator accepts the exact positive boundary");
  test.expect(!within_bound(std::nextafter(0.25, std::numeric_limits<double>::infinity()),
                            0.0, 0.25),
              "inclusive comparator rejects the next double outside the boundary");
  test.expect(within_bound(-0.0, 0.0, 0.0),
              "inclusive comparator accepts equal signed zeros with zero budget");

  test.expect(std::isnan(error_bound(0, 1.0)) && std::isnan(error_bound(17, 1.0)) &&
                  std::isnan(error_bound(1040, 1.0)),
              "invalid K values return quiet NaN");
  test.expect(std::isnan(error_bound(16, -1.0)) && std::isnan(error_bound(16, nan)) &&
                  std::isnan(error_bound(16, infinity)) &&
                  std::isnan(error_bound(16, -infinity)),
              "negative and nonfinite scales return quiet NaN");
  test.expect(std::isfinite(error_bound(1024, std::numeric_limits<double>::max())) &&
                  error_bound(1024, std::numeric_limits<double>::max()) > 0.0,
              "largest finite scale has a finite positive budget");

  test.expect(!within_bound(nan, 0.0, 0.0) && !within_bound(0.0, nan, 0.0) &&
                  !within_bound(0.0, 0.0, nan),
              "NaN in any comparator argument is rejected");
  test.expect(!within_bound(infinity, 0.0, 0.0) && !within_bound(0.0, infinity, 0.0) &&
                  !within_bound(0.0, 0.0, infinity),
              "infinity in any comparator argument is rejected");
  test.expect(!within_bound(0.0, 0.0, -0.25),
              "negative comparator budget is rejected");
  return test.result();
}
