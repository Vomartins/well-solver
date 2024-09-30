#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
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

  std::cout << "Dvals: ";
  std::cout << size(Dvals) << " " <<std::endl;
  //for (const auto& val : Dvals) std::cout << val << " ";
  //std::cout << std::endl;
  std::cout << "Dcols: ";
  std::cout << size(Dcols) << " " <<std::endl;
  for (const auto& val : Dcols) std::cout << val << " ";
  std::cout << std::endl;
  std::cout << "Drows: ";
  std::cout << size(Drows) << " " <<std::endl;
  //for (const auto& val : Drows) std::cout << val << " ";
  //std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Bvals: ";
  std::cout << size(Bvals) << " " <<std::endl;
  std::cout << "Bcols: ";
  std::cout << size(Bcols) << " " <<std::endl;
  std::cout << "Brows: ";
  std::cout << size(Brows) << " " <<std::endl;
  for (const auto& val : Bcols) std::cout << val << " ";
  std::cout << std::endl;

  std::cout << "Cvals: ";
  std::cout << size(Cvals) << " " <<std::endl;
  std::cout << "Ccols: ";
  std::cout << size(Ccols) << " " <<std::endl;
  std::cout << "Crows: ";
  std::cout << size(Crows) << " " <<std::endl;
  for (const auto& val : Ccols) std::cout << val << " ";
  std::cout << std::endl;

  std::vector<double> vecRes = loadResVector("vector-Res.bin");

  std::cout << "vecRes: ";
  std::cout << size(vecRes) << " " <<std::endl;
  //for (const auto& val : vecRes) std::cout << val << " ";
  //std::cout << std::endl;

  unsigned int Mb = 16;
  unsigned int length = 4*Mb;
  std::vector<double> z1(length, 0.0);          // z1 = B * x
  std::vector<double> z2(length, 0.0);
  //std::fill(z1.begin(), z1.end(), 0.0);
  //std::fill(z2.begin(), z2.end(), 0.0);

  for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim_wells; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim; ++k){
                    //std::cout << colIdx * dim + k << std::endl;
                    //std::cout << temp << std::endl;
                }
                z1[row * dim_wells + j] += temp;
                //std::cout << z1[row * dim_wells + j] << " ";
            }
        }
    }

/*
  for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim_wells; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim; ++k) {
                    std::cout << Bvals[blockID * dim * dim_wells + j * dim + k] << " --- " << std::endl;
                    temp += Bvals[blockID * dim * dim_wells + j * dim + k] * vecRes[colIdx * dim + k];
                    //std::cout << temp << std::endl;
                }
                z1[row * dim_wells + j] += temp;
                //std::cout << z1[row * dim_wells + j] << " ";
            }
        }
    }
*/

    // z1 = vecRes;

    std::cout << std::endl;
    std::cout << "\nBx: ";
    std::cout << size(z1) << std::endl;
    for (const auto& val : z1) std::cout << val << " ";
    std::cout << std::endl;

    unsigned int M = dim_wells*Mb;
    void *UMFPACK_Symbolic, *UMFPACK_Numeric;

    (void) umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    (void) umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);
    (void) umfpack_di_solve(UMFPACK_A, Dcols.data(), Drows.data(), Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);

    std::cout << std::endl;
    std::cout << "\nD-1Bx: ";
    std::cout << size(z2) << std::endl;
    for (const auto& val : z2) std::cout << val << " ";
    std::cout << std::endl;

    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);
}
