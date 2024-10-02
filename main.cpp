#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include <algorithm>
#include <math.h>
#include "rocsparse-ilu0.hpp"
//#include "well-matrices.hpp"
#include "read-vectors.hpp"

//#include <dune/istl/umfpack.hh>
#include <umfpack.h>

double errorInfinityNorm(std::vector<double> vecSol, std::vector<double> vec, bool printError = false, bool printErrorVector = false){
  std::vector<double> error(size(vecSol), -1.0);

  for (int i=0; i<size(vec); i++){
    error[i]= fabs(vec[i]-vecSol[i]);
  }

  size_t norm_error = *std::max_element(error.begin(),error.end());

  if (printErrorVector) {
        std::cout << "Error vector: [ ";
        for (const auto& e : error) {
            std::cout << e << " ";
        }
        std::cout << "]" << std::endl;
    }

  if (printError) std::cout << "Inifinity norm of error: " << norm_error << std::endl;

  return norm_error;
}

int main(int argc, char ** argv)
{

  const static int dim = 3;
  const static int dim_wells = 4;
/*
  WellMatrices wellMatrices;

  wellMatrices.read_matrices();

  // std::cout << wellMatrices.duneB[0][0] << std::endl;


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

  std::cout << wellMatrices.duneD.nonzeroes() << std::endl;
  std::cout << wellMatrices.duneD.N() << std::endl;
  std::cout << wellMatrices.duneD.M() << std::endl;

  //std::cout << wellMatrices.duneB.nonzeroes() << std::endl;
  //std::cout << wellMatrices.duneB.N() << std::endl;
  //std::cout << wellMatrices.duneB.M() << std::endl;

  //std::cout << wellMatrices.duneC.nonzeroes() << std::endl;
  //std::cout << wellMatrices.duneC.N() << std::endl;
  //std::cout << wellMatrices.duneC.M() << std::endl;
*/
  std::cout << "########## std::vector ##########" << std::endl;

  //std::string reservoir = argv[1];

  //std::cout << reservoir <<std::endl;

  std::vector<double> Dvals, Bvals, Cvals;
  std::vector<int> Dcols, Drows;
  std::vector<unsigned int> Bcols, Ccols, Brows, Crows;

  loadSparseMatrixVectors(Dvals, Dcols, Drows, "data/matrix-D.bin");
  loadSparseMatrixVectors(Bvals, Bcols, Brows, "data/matrix-B.bin");
  loadSparseMatrixVectors(Cvals, Ccols, Crows, "data/matrix-C.bin");
  std::vector<double> vecRes = loadResVector("data/vector-Res.bin");
  std::vector<double> vecSol = loadResVector("data/vector-Sol.bin");

  //std::cout << "Dvals: " << size(Dvals) << std::endl;
  //for (const auto& val : Dvals) std::cout << val << " ";
  //std::cout << std::endl;

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
    std::cout << std::endl;

    unsigned int M = dim_wells*Mb;
    void *UMFPACK_Symbolic, *UMFPACK_Numeric;
    std::cout << "########## UMFpack Solver ########## " << std::endl;

    umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);
    umfpack_di_solve(UMFPACK_A, Dcols.data(), Drows.data(), Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);

    std::cout << "\nD-1Bx: ";
    std::cout << size(z2) << std::endl;
    for (const auto& val : z2) std::cout << val << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    double errorCPU = errorInfinityNorm(vecSol, z2, true, true);
    std::cout << std::endl;


    //Dvals = {5, 8, 6, 3};
    //Drows = {0, 1, 3, 2};
    //Dcols = {0, 1, 3, 4, 4};

    //Dvals = {3, 4, 7, 1, 5, 2, 9, 6, 5};
    //Drows = {0, 1, 2, 0, 2, 0, 2, 4, 4};
    //Dcols = {0, 1, 3, 5, 8, 9};

    //Dvals = {1, 6, 2, 8, 7, 3, 9, 4, 5};
    //Drows = {0, 2, 1, 3, 0, 2, 1, 3, 4};
    //Dcols = {0, 2, 4, 6, 8, 9};

    std::cout << "########## RocSPARSE Solver ########## " << std::endl;

    unsigned int nnzs = size(Dvals);
    //std::cout << "nnzs: " << nnzs << std::endl;
    unsigned int sizeDvals = size(Dvals);
    //std::cout << "sizeDvals: " << sizeDvals <<std::endl;
    //for(const auto& val : Dvals) std::cout << val << " ";
    //std::cout << std::endl;
    unsigned int sizeDcols = size(Dcols);
    //std::cout << "sizeDcols: " << sizeDcols <<std::endl;
    //for(const auto& val : Dcols) std::cout << val << " ";
    //std::cout << std::endl;
    unsigned int sizeDrows = size(Drows);
    //std::cout << "sizeDrows: " << sizeDrows <<std::endl;
    //for(const auto& val : Drows) std::cout << val << " ";
    //std::cout << std::endl;

    std::vector<double> Dvals_(sizeDvals);
    std::vector<int> Dcols_(sizeDvals);
    std::vector<int> Drows_(sizeDcols);

    squareCSCtoCSR(Dvals, Drows, Dcols, Dvals_, Drows_, Dcols_);

    //std::cout << "############ Conversion ############" << std::endl;
    //std::cout << "Dcols[-1] nnzb :" << Dcols[sizeDcols-1] << std::endl;
    //std::cout << "############ Dvals_ ############" << std::endl;
    unsigned int sizeDvals_ = size(Dvals_);
    //std::cout << "sizeDvals_: " << sizeDvals_ << std::endl;
    //for(const auto& val : Dvals_) std::cout << val << " ";
    //std::cout << std::endl;
    //std::cout << "############ Drows_ ############" << std::endl;
    unsigned int sizeDrows_ = size(Drows_);
    //std::cout << "sizeDrows_: " << sizeDrows_ << std::endl;
    //for(const auto& val : Drows_) std::cout << val << " ";
    //std::cout << std::endl;
    //std::cout << "############ Dcols_ ############" << std::endl;
    unsigned int sizeDcols_ = size(Dcols_);
    //std::cout << "sizeDcols_: " << sizeDcols_ << std::endl;
    //for(const auto& val : Dcols_) std::cout << val << " ";
    //std::cout << std::endl;


    M = size(Drows_)-1 ;

    MultisegmentWellContribution mswc;
    mswc.initialize(M, nnzs, sizeDvals_, sizeDrows_, sizeDcols_);
    mswc.copyHostToDevice(Dvals_, Drows_, Dcols_, z1);

    std::vector<double> z2_rocsparse = mswc.solveSytem();
    std::cout << "\nD-1Bx: ";
    std::cout << size(z2_rocsparse) << std::endl;
    for(double val : z2_rocsparse) std::cout << val << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    double errorGPURocSPARSE = errorInfinityNorm(vecSol, z2_rocsparse, true, true);
    std::cout << std::endl;

    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);

}
