#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <algorithm>
//#include "rocsparse-ilu0.hpp"
//#include "well-matrices.hpp"
#include <dune/common/version.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixmarket.hh>

#include "read-vectors.hpp"

//#include <dune/istl/umfpack.hh>
#include <umfpack.h>

int main(int argc, char ** argv)
{

  //WellMatrices wellMatrices;

  //wellMatrices.read_matrices();

  // std::cout << wellMatrices.duneB[0][0] << std::endl;

  const static int dim = 3;
  const static int dim_wells = 4;
/*
  // Matrix type for matrix D
  using DiagMatrixBlockWellType = Dune::FieldMatrix<double,dim_wells,dim_wells>;
  using DiagMatWell = Dune::BCRSMatrix<DiagMatrixBlockWellType>;

  // Matrix type for matrices D and C^t
  using OffDiagMatrixBlockWellType = Dune::FieldMatrix<double,dim_wells,dim>;
  using OffDiagMatWell = Dune::BCRSMatrix<OffDiagMatrixBlockWellType>;

  DiagMatWell duneD;
  OffDiagMatWell duneB;
  OffDiagMatWell duneC;

  const char* fileName_D = "matrix-D.mm";
  const char* fileName_B = "matrix-B.mm";
  const char* fileName_C = "matrix-C.mm";

  std::ifstream fileIn_D(fileName_D);
  std::ifstream fileIn_B(fileName_B);
  std::ifstream fileIn_C(fileName_C);

  Dune::readMatrixMarket(duneD, fileIn_D);
  Dune::readMatrixMarket(duneB, fileIn_B);
  Dune::readMatrixMarket(duneC, fileIn_C);

  std::cout << "########## Dune Matrices ##########" << std::endl;

  std::cout << duneD.nonzeroes() << std::endl;
  std::cout << duneD.N() << std::endl;
  std::cout << duneD.M() << std::endl;

  std::cout << duneB.nonzeroes() << std::endl;
  std::cout << duneB.N() << std::endl;
  std::cout << duneB.M() << std::endl;

  std::cout << duneC.nonzeroes() << std::endl;
  std::cout << duneC.N() << std::endl;
  std::cout << duneC.M() << std::endl;
*/
  std::cout << "########## std::vector ##########" << std::endl;

  std::vector<double> Dvals, Bvals, Cvals;
  std::vector<int> Dcols, Drows;
  std::vector<unsigned int> Bcols, Ccols, Brows, Crows;

  loadBCRSMatrixVectors(Dvals, Dcols, Drows, "matrix-D.bin");
  loadBCRSMatrixVectors(Bvals, Bcols, Brows, "matrix-B.bin");
  loadBCRSMatrixVectors(Cvals, Ccols, Crows, "matrix-C.bin");

  std::vector<double> vecRes = loadResVector("vector-Res.bin");


  unsigned int Mb = size(Brows)-1 ;
  unsigned int length = dim_wells*Mb;
  std::vector<double> z1(length, 0.0);          // z1 = B * x
  std::vector<double> z2(length, 0.0);

  // B*x multiplication
  for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim_wells; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim; ++k) {
                    temp += Bvals[blockID * dim * dim_wells + j * dim + k] * vecRes[colIdx * dim + k];
                }
                z1[row * dim_wells + j] += temp;
            }
        }
    }

    std::cout << "\nBx: ";
    std::cout << size(z1) << std::endl;
    for (const auto& val : z1) std::cout << val << " ";
    std::cout << std::endl;

    unsigned int M = dim_wells*Mb;
    void *UMFPACK_Symbolic, *UMFPACK_Numeric;

    umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);
    umfpack_di_solve(UMFPACK_A, Dcols.data(), Drows.data(), Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);

    std::cout << "\nD-1Bx: ";
    std::cout << size(z2) << std::endl;
    for (const auto& val : z2) std::cout << val << " ";
    std::cout << std::endl;

    std::vector<double> vecSol = loadResVector("vector-Sol.bin");
    std::vector<double> error(size(vecSol), -1.0);
    for (int i=0; i<size(z2); i++){
      error[i]= abs(z2[i]-vecSol[i]);
      //std::cout << error[i] << " ";
    }
    std::cout << std::endl;
    size_t max_error = *std::max_element(error.begin(),error.end());
    std::cout << "Inifinity norm of error: " << max_error << std::endl;

    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);
}
