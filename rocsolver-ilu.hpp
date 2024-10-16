#ifndef ROCSOLVER_ILU_HPP
#define ROCSOLVER_ILU_HPP

#include <iostream>
#include <vector>

#include <hip/hip_runtime_api.h>
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
  int sizeBvals;
  int sizeBcols;
  int sizeBrows;
  unsigned int resSize;
  unsigned int CzSize;
  rocblas_int *info;
  rocblas_int *ipiv;
  double *d_Dmatrix_hip;
  double *d_Cvals_hip;
  double *d_Bvals_hip;
  unsigned int *d_Brows_hip;
  unsigned int *d_Bcols_hip;
  void *d_buffer;
  rocblas_handle handle;
  //rocsparse_mat_descr descr_A, descr_M, descr_L, descr_U;
  rocsolver_rfinfo ilu_info;
  rocblas_operation operation = rocblas_operation_none;
  double *x_hip;
  double *z_hip;
  double *y_hip;
  double *sol_hip;
  double *rhs_hip;
  std::vector<double> vecSol;
  std::vector<double> z1;
  std::vector<double> vecY;

public:
  ~RocsolverMSWContribution();
  void initialize(rocblas_int M,
                  rocblas_int N,
                  int sizeBvals,
                  int sizeBcols,
                  int sizeBrows,
                  int resSize,
                  int CzSize);
  void copyHostToDevice(double *Dmatrix,
                        std::vector<double> Cvals,
                        std::vector<double> Bvals,
                        std::vector<unsigned int> Brows,
                        std::vector<unsigned int> Bcols,
                        std::vector<double> x,
                        std::vector<double> y);
  std::vector<double> solveSytem();
  void scalar_csr(int m,
                int threads_per_block,
                unsigned int * row_offsets,
                unsigned int * cols,
                double * vals,
                double * x,
                double * y,
                double alpha,
                double beta);
  void scalar_csc(int n,
                  int threads_per_block,
                  unsigned int* col_offsets,
                  unsigned int* rows,
                  double* vals,
                  double* x,
                  double* y,
                  double alpha,
                  double beta);
  void spmv_k(int n,
              int threads_per_block,
              unsigned int* row_offsets,
              unsigned int* cols,
              double* vals,
              double* x,
              double* y);
  void blockrsmvBx(double* vals,
                        unsigned int* cols,
                        unsigned int* rows,
                        double* x,
                        double* rhs,
                        double* out,
                        int Nb,
                        unsigned int block_dimM,
                        unsigned int block_dimN,
                        const double op_sign);
  /*void blocksrmvCtz(double* vals,
                    unsigned int* cols,
                    unsigned int* rows,
                    double* x,
                    double* rhs,
                    double* out,
                    unsigned int Nb,
                    unsigned int block_dimM,
                    unsigned int block_dimN,
                    const double op_sign,
                    const unsigned int resSize,
                    const int sizeBvals);*/
  void blocksrmvC_z(double* vals, unsigned int* cols, unsigned int* rows, double* z, double* y, unsigned int Nb, unsigned int block_dimM, unsigned int block_dimN);

  std::vector<double> apply();

  std::vector<double> vectorCtz();
};

#endif
