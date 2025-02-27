#include "mathprim/optim/ex_probs/quad.hpp"
#include <iostream>

using namespace mathprim;

int main() {
  optim::ex_probs::quad_problem<double> problem(10);
  problem.setup();
  std::cout << problem.eval_value() << std::endl;
  return 0;
}
