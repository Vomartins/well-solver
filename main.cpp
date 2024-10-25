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

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>


#define VAR_NAME(var) (#var)

__global__ void test(double a, double b, double res){
  if (threadIdx.x == 0)
    printf("a: %f\n", a);
    printf("b: %f\n", b);
    res = a - b;
}

double errorInfinityNorm(std::vector<double> vecSol, std::vector<double> vec, bool printError = false, bool printErrorVector = false){
  std::vector<double> error(size(vecSol), -1.0);
  if (size(vecSol) == size(vec)){
    for (int i=0; i<size(vec); i++){
      error[i]= fabs(vec[i]-vecSol[i]);
    }

    double norm_error = *std::max_element(error.begin(),error.end());

    if (printErrorVector) {
          std::cout << "Error vector: [ ";
          for (const auto& e : error) {
              std::cout << e << " ";
          }
          std::cout << "]" << std::endl;
      }

    if (printError) std::cout << "Inifinity norm of error: " << norm_error << std::endl;

    return norm_error;
  } else { return -1.0;}
}

void errorInfinityNormDebug(std::vector<double> vecSol, std::vector<double> vec, double epsilon, bool printError = false, bool printErrorVector = false){
  std::vector<double> error(size(vecSol), -1.0);
  if (size(vecSol) == size(vec)){
    for (int i=0; i<size(vec); i++){
      error[i]= fabs(vec[i]-vecSol[i]);
      if (error[i] > epsilon){
        std::cout << "Element " << i << " has an error of " << error[i] << std::endl;
      }
    }

    double norm_error = *std::max_element(error.begin(),error.end());

    if (printErrorVector) {
          std::cout << "Error vector: [ ";
          for (const auto& e : error) {
              std::cout << e << " ";
          }
          std::cout << "]" << std::endl;
      }

    if (printError) std::cout << "Inifinity norm of error: " << norm_error << std::endl;


  }
}

double sumVector(std::vector<double> vec){
  double sum = 0.0;
  for (const auto& val : vec) sum += val;
  return sum;
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
  std::cout << std::endl;

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
  //std::cout << "########## std::vector ##########" << std::endl;

  std::vector<double> Dvals, Bvals, Cvals;
  std::vector<int> Dcols, Drows;
  std::vector<unsigned int> Bcols, Ccols, Brows, Crows;

  loadSparseMatrixVectors(Dvals, Dcols, Drows, "data/"+reservoir+"/matrix-D.bin");
  loadSparseMatrixVectors(Bvals, Bcols, Brows, "data/"+reservoir+"/matrix-B.bin");
  loadSparseMatrixVectors(Cvals, Ccols, Crows, "data/"+reservoir+"/matrix-C.bin");
  std::vector<double> vecRes = loadResVector("data/"+reservoir+"/vector-Res.bin");
  std::vector<double> vecSol = loadResVector("data/"+reservoir+"/vector-Sol.bin");

  //std::cout << size(vecRes) << std::endl;
  //PRINT_VECTOR(Bvals);
  //PRINT_VECTOR(Bcols);
  //PRINT_VECTOR(Brows);

  unsigned int Mb = size(Brows)-1 ;
  unsigned int length = dim_wells*Mb;

  std::vector<double> z1(length, 0.0);          // z1 = B * x
  std::vector<double> z2(length, 0.0);

  Dune::Timer cpuMatrix;
  cpuMatrix.start();
  // B*x multiplication
  for (unsigned int row = 0; row < Mb; ++row) {
        // for every block in the row
        for (unsigned int blockID = Brows[row]; blockID < Brows[row + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim_wells; ++j) {
                double temp = 0.0;
                for (unsigned int k = 0; k < dim; ++k) {
                    double B_elem = Bvals[blockID * dim * dim_wells + j * dim + k];
                    double x_elem = vecRes[colIdx * dim + k];
                    temp += Bvals[blockID * dim * dim_wells + j * dim + k] * vecRes[colIdx * dim + k];
                    //printf("B_elem: %.15f(%u), x_elem: %.15f(%u), local_out: %.15f\n", B_elem, blockID * dim * dim_wells + j * dim + k, x_elem, colIdx * dim + k, temp);
                }
                z1[row * dim_wells + j] += temp;
                //printf("y_elem: %.15f(%u)\n", z1[row * dim_wells + j], row * dim_wells + j);
            }
        }
    }
    cpuMatrix.stop();
    double cpuTime = cpuMatrix.lastElapsed();

    unsigned int M = dim_wells*Mb;
    void *UMFPACK_Symbolic, *UMFPACK_Numeric;
    std::cout << "########## UMFPack Solver ########## " << std::endl;
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

    //PRINT_VECTOR(z2);

    PRINT_VECTOR(UMFPackTimes);

    double UMFPackTotalTime = sumVector(UMFPackTimes);

    std::cout << "Total time: "<< UMFPackTotalTime+cpuTime << "s" << std::endl;

    double errorCPU = errorInfinityNorm(vecSol, z2, true);
    std::cout << std::endl;

    umfpack_di_free_symbolic(&UMFPACK_Symbolic);
    umfpack_di_free_numeric(&UMFPACK_Numeric);
/*
    std::cout << "########## RocSPARSE Solver ########## " << std::endl;
    Dune::Timer RocSPARSETimer;
*/
    unsigned int nnzs = size(Dvals);
    unsigned int sizeDvals = size(Dvals);
    unsigned int sizeDcols = size(Dcols);
    unsigned int sizeDrows = size(Drows);
/*
    std::vector<double> Dvals_(sizeDvals);
    std::vector<int> Dcols_(sizeDvals);
    std::vector<int> Drows_(sizeDcols);

    RocSPARSETimer.start();
    squareCSCtoCSR(Dvals, Drows, Dcols, Dvals_, Drows_, Dcols_);
    RocSPARSETimer.stop();
    RocSPARSETimes[0] += RocSPARSETimer.lastElapsed();

    unsigned int sizeDvals_ = size(Dvals_);
    unsigned int sizeDrows_ = size(Drows_);
    unsigned int sizeDcols_ = size(Dcols_);

    M = size(Drows_)-1 ;

    RocsparseMSWContribution rocsparseMswc;

    RocSPARSETimer.start();
    rocsparseMswc.initialize(M, nnzs, sizeDvals_, sizeDrows_, sizeDcols_);
    RocSPARSETimer.stop();
    RocSPARSETimes[0] += RocSPARSETimer.lastElapsed();

    RocSPARSETimer.start();
    rocsparseMswc.copyHostToDevice(Dvals_, Drows_, Dcols_, z1);
    RocSPARSETimer.stop();
    RocSPARSETimes[1] += RocSPARSETimer.lastElapsed();

    RocSPARSETimer.start();
    std::vector<double> z2_rocsparse = rocsparseMswc.solveSytem();
    RocSPARSETimer.stop();
    RocSPARSETimes[2] += RocSPARSETimer.lastElapsed();

    //PRINT_VECTOR(z2_rocsparse);

    PRINT_VECTOR(RocSPARSETimes);

    double RocSparseTotalTime = sumVector(RocSPARSETimes);

    std::cout << "Total time: "<< RocSparseTotalTime+cpuTime << "s" << std::endl;

    double errorGPURocSPARSE = errorInfinityNorm(vecSol, z2_rocsparse, true);
    std::cout << std::endl;
*/
    std::cout << "########## RocSOLVER Solver ########## " << std::endl;
    Dune::Timer RocSOLVERTimer;

    std::vector<double> vecy = loadResVector("data/"+reservoir+"/vector-y.bin");
    //std::vector<double> vecY = loadResVector("data/"+reservoir+"/vector-Y.bin");

    int lda = sizeDcols-1;
    int sizeBvals = size(Bvals);
    int sizeBrows = size(Brows);
    int sizeBcols = size(Bcols);
    int resSize = size(vecRes);
    int CzSize = size(vecy);

    RocSOLVERTimer.start();
    double* Dmatrix = squareCSCtoMatrix(Dvals, Drows, Dcols);
    RocSOLVERTimer.stop();
    RocSOLVERTimes[0] += RocSOLVERTimer.lastElapsed();

    RocsolverMSWContribution rocsolverMswc;

    RocSOLVERTimer.start();
    rocsolverMswc.initialize(lda, lda, sizeBvals, sizeBcols, sizeBrows, resSize, CzSize);
    RocSOLVERTimer.stop();
    RocSOLVERTimes[1] += RocSOLVERTimer.lastElapsed();

    RocSOLVERTimer.start();
    rocsolverMswc.copyHostToDevice(Dmatrix, Cvals, Bvals, Brows, Bcols, vecRes, vecy);
    RocSOLVERTimer.stop();
    RocSOLVERTimes[1] += RocSOLVERTimer.lastElapsed();

    RocSOLVERTimer.start();
    std::vector<double> z2_rocsolver = rocsolverMswc.apply();
    RocSOLVERTimer.stop();
    RocSOLVERTimes[2] += RocSOLVERTimer.lastElapsed();


    PRINT_VECTOR(RocSOLVERTimes);

    double RocSolverTotalTime = (RocSOLVERTimes[0]+RocSOLVERTimes[2]);

    std::cout << "Total time: "<< RocSolverTotalTime << "s" << std::endl;

    double errorGPURocSOLVER = errorInfinityNorm(vecSol, z2_rocsolver, true);
    std::cout << std::endl;

    std::vector<double> vecY_hip = rocsolverMswc.vectorCtz();
    int row = 0;
    cpuMatrix.start();
    // y -= (C^T * z2)
    // y -= (C^T * (D^-1 * (B * x)))
    for (unsigned int blockrow = 0; blockrow < Mb; ++blockrow) {
        // for every block in the row
        for (unsigned int blockID = Brows[blockrow]; blockID < Brows[blockrow + 1]; ++blockID) {
            unsigned int colIdx = Bcols[blockID];
            for (unsigned int j = 0; j < dim; ++j) {
                double temp = 0.0;
                row = colIdx * dim + j;
                //printf("Row: %u,\n", row);
                for (unsigned int k = 0; k < dim_wells; ++k) {
                    double C_elem = Cvals[blockID * dim * dim_wells + j + k * dim];
                    double z_elem = z2[blockrow * dim_wells + k];
                    //temp += C_elem * z_elem;
                    vecy[row] -= C_elem * z_elem;
                    //printf("C_elem: %.15f(%u), z_elem: %.15f(%u), local_out: %.15f, y(row): %.15f\n", C_elem, blockID * dim * dim_wells + j + k * dim, z_elem, blockrow * dim_wells + k, C_elem * z_elem, vecy[row]);
                }
                //printf("block: %i, col: %i\n", blockID, colIdx);
                //printf("y_elem: %i\n", colIdx * dim + j);

                //vecy[row] -= temp;
                //printf("y(row): %.12f\n", vecy[row]);
            }
        }
    }
    std::cout << std::endl;
    cpuMatrix.stop();
    cpuTime += cpuMatrix.lastElapsed();

    std::cout << "Well contribution" << std::endl;
    errorInfinityNormDebug(vecY_hip, vecy, 1e-10, true);
    std::cout << std::endl;

    std::cout << "########## Speed Factors ########## " << std::endl;
    //std::cout << "RocSPARSE: " << (RocSparseTotalTime+cpuTime)/(UMFPackTotalTime+cpuTime) << std::endl;
    std::cout << "RocSOLVER: " << (UMFPackTotalTime+cpuTime)/RocSolverTotalTime << std::endl;
    std::cout << std::endl;
}
