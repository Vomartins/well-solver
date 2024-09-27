 #include "well-matrices.hpp"

 void WellMatrices::read_matrices()
  {
    Dune::loadMatrixMarket(duneD, "matrix-D.mm");
    Dune::loadMatrixMarket(duneB, "matrix-B.mm");
    Dune::loadMatrixMarket(duneC, "matrix-C.mm");
  }
