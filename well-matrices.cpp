 #include "well-matrices.hpp"

 void WellMatrices::read_matrices()
  {
    Dune::loadMatrixMarket(duneD, fileName_D);
    //Dune::loadMatrixMarket(duneB, fileName_B);
    //Dune::loadMatrixMarket(duneC, fileName_C);

    //Dune::readMatrixMarket(duneD, fileIn_D);
    //Dune::readMatrixMarket(duneB, fileIn_B);
    //Dune::readMatrixMarket(duneC, fileIn_C);
  }
