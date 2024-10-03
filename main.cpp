#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>
#include <algorithm>
#include <math.h>
#include "read-vectors.hpp"
#include "rocsparse-ilu0.hpp"
#include "rocsolver-ilu.hpp"
#include <umfpack.h>

#include <dune/common/timer.hh>

#define VAR_NAME(var) (#var)

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

template <typename T>
void printVector(const std::vector<T>& vec, const std::string& name){
  std::cout << name <<": " << size(vec) << std::endl;
  std::cout << "[ ";
  for (const auto& val : vec) std::cout << val << " ";
  std::cout << "]";
  std::cout << std::endl;
  std::cout << std::endl;
}

template <typename T>
void printVectorWrapper(const std::vector<T>& vec, const std::string& name) {
    printVector(vec, name);
}

#define PRINT_VECTOR(vec) printVectorWrapper(vec, VAR_NAME(vec))

int main(int argc, char ** argv)
{

  std::vector<double> UMFPackTimes(3, 0.0);
  std::vector<double> RocSPARSETimes(3, 0.0);
  std::vector<double> RocSOLVERTimes(3, 0.0);

  const static int dim = 3;
  const static int dim_wells = 4;

  std::string reservoir;

  if (argc > 1){
    reservoir = argv[1];

    std::cout << "Reservoir: " << reservoir << std::endl;
  } else {
    std::cout << "No reservoir provided. Please enter one option as command-line argument (norne, msw)." << std::endl;
    return 1;
  }

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

  std::vector<double> Dvals, Bvals, Cvals;
  std::vector<int> Dcols, Drows;
  std::vector<unsigned int> Bcols, Ccols, Brows, Crows;

  loadSparseMatrixVectors(Dvals, Dcols, Drows, "data/"+reservoir+"/matrix-D.bin");
  loadSparseMatrixVectors(Bvals, Bcols, Brows, "data/"+reservoir+"/matrix-B.bin");
  loadSparseMatrixVectors(Cvals, Ccols, Crows, "data/"+reservoir+"/matrix-C.bin");
  std::vector<double> vecRes = loadResVector("data/"+reservoir+"/vector-Res.bin");
  std::vector<double> vecSol = loadResVector("data/"+reservoir+"/vector-Sol.bin");

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

    PRINT_VECTOR(z1);

    unsigned int M = dim_wells*Mb;
    void *UMFPACK_Symbolic, *UMFPACK_Numeric;
    std::cout << "########## UMFpack Solver ########## " << std::endl;
    Dune::Timer UMFPackTimer;

    UMFPackTimer.start();
    umfpack_di_symbolic(M, M, Dcols.data(), Drows.data(), Dvals.data(), &UMFPACK_Symbolic, nullptr, nullptr);
    umfpack_di_numeric(Dcols.data(), Drows.data(), Dvals.data(), UMFPACK_Symbolic, &UMFPACK_Numeric, nullptr, nullptr);
    UMFPackTimer.stop();
    UMFPackTimes[0] = UMFPackTimer.lastElapsed();

    UMFPackTimer.start();
    umfpack_di_solve(UMFPACK_A, Dcols.data(), Drows.data(), Dvals.data(), z2.data(), z1.data(), UMFPACK_Numeric, nullptr, nullptr);
    UMFPackTimer.stop();
    UMFPackTimes[2] = UMFPackTimer.lastElapsed();

    PRINT_VECTOR(z2);

    PRINT_VECTOR(UMFPackTimes);

    double errorCPU = errorInfinityNorm(vecSol, z2, true);
    std::cout << std::endl;

    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);

    std::cout << "########## RocSPARSE Solver ########## " << std::endl;
    Dune::Timer RocSPARSETimer;

    unsigned int nnzs = size(Dvals);
    unsigned int sizeDvals = size(Dvals);
    unsigned int sizeDcols = size(Dcols);
    unsigned int sizeDrows = size(Drows);

    std::vector<double> Dvals_(sizeDvals);
    std::vector<int> Dcols_(sizeDvals);
    std::vector<int> Drows_(sizeDcols);

    RocSPARSETimer.start();
    squareCSCtoCSR(Dvals, Drows, Dcols, Dvals_, Drows_, Dcols_);
    RocSPARSETimer.stop();
    RocSOLVERTimes[0] += RocSPARSETimer.lastElapsed();

    unsigned int sizeDvals_ = size(Dvals_);
    unsigned int sizeDrows_ = size(Drows_);
    unsigned int sizeDcols_ = size(Dcols_);

    M = size(Drows_)-1 ;

    RocsparseMSWContribution rocsparseMswc;

    RocSPARSETimer.start();
    rocsparseMswc.initialize(M, nnzs, sizeDvals_, sizeDrows_, sizeDcols_);
    RocSPARSETimer.stop();
    RocSOLVERTimes[0] += RocSPARSETimer.lastElapsed();

    RocSPARSETimer.start();
    rocsparseMswc.copyHostToDevice(Dvals_, Drows_, Dcols_, z1);
    RocSPARSETimer.stop();
    RocSOLVERTimes[1] += RocSPARSETimer.lastElapsed();

    RocSPARSETimer.start();
    std::vector<double> z2_rocsparse = rocsparseMswc.solveSytem();
    RocSPARSETimer.stop();
    RocSOLVERTimes[2] += RocSPARSETimer.lastElapsed();

    PRINT_VECTOR(z2_rocsparse);

    PRINT_VECTOR(RocSOLVERTimes);

    double errorGPURocSPARSE = errorInfinityNorm(vecSol, z2_rocsparse, true);
    std::cout << std::endl;

    std::cout << "########## RocSOLVER Solver ########## " << std::endl;
    Dune::Timer RocSOLVERTimer;

    int lda = sizeDcols-1;

    RocSOLVERTimer.start();
    double* Dmatrix = squareCSCtoMatrix(Dvals, Drows, Dcols);
    RocSOLVERTimer.stop();
    RocSOLVERTimes[0] += RocSOLVERTimer.lastElapsed();

    RocsolverMSWContribution rocsolverMswc;

    RocSOLVERTimer.start();
    rocsolverMswc.initialize(lda, lda);
    RocSOLVERTimer.stop();
    RocSOLVERTimes[0] += RocSOLVERTimer.lastElapsed();

    RocSOLVERTimer.start();
    rocsolverMswc.copyHostToDevice(Dmatrix, z1);
    RocSOLVERTimer.stop();
    RocSOLVERTimes[1] += RocSOLVERTimer.lastElapsed();

    RocSOLVERTimer.start();
    std::vector<double> z2_rocsolver = rocsolverMswc.solveSytem();
    RocSOLVERTimer.stop();
    RocSOLVERTimes[2] += RocSOLVERTimer.lastElapsed();

    PRINT_VECTOR(z2_rocsolver);

    PRINT_VECTOR(RocSOLVERTimes);

    double errorGPURocSOLVER = errorInfinityNorm(vecSol, z2_rocsolver, true);
    std::cout << std::endl;


}
