#ifndef ROCSOLVER_ILU_HPP
#define ROCSOLVER_ILU_HPP

#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <rocsolver/rocsolver.h>

class RocsolverMSWContribution
{
private:
  bool analysis_done = false;
  rocblas_int M;
  rocblas_int N;
  rocblas_int Nrhs = 1;
  rocblas_int lda;
  rocblas_int ldb;
  rocblas_int *info;
  rocblas_int *ipiv;
  double *d_Dmatrix_hip;
  void *d_buffer;
  rocblas_handle handle;
  //rocsparse_mat_descr descr_A, descr_M, descr_L, descr_U;
  rocsolver_rfinfo ilu_info;
  rocblas_operation operation = rocblas_operation_none;
  double *z_hip;
  std::vector<double> vecSol;

public:
  ~RocsolverMSWContribution();
  void initialize(rocblas_int M, rocblas_int N);
  void copyHostToDevice(double *Dmatrix, std::vector<double> z1);
  std::vector<double> solveSytem();
};

#endif
