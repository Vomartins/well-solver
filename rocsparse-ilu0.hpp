#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <rocsparse/rocsparse.h>

class MultisegmentWellContribution
{

private:
  bool analysis_done = false;
  unsigned int sizeDvals;
  unsigned int sizeDcols;
  unsigned int sizeDrows;
  int nnzs;
  rocsparse_int M;
  rocsparse_int *d_Drows_hip;
  rocsparse_int *d_Dcols_hip;
  double *d_Dvals_hip;
  void *d_buffer;
  rocsparse_handle handle;
  rocsparse_mat_descr descr_A, descr_M, descr_L, descr_U;
  rocsparse_mat_info ilu_info;
  rocsparse_operation operation = rocsparse_operation_none;
  double *z_aux_hip;
  double *z1_hip;
  double *z2_hip;
  std::vector<double> vecSol;

public:
  ~MultisegmentWellContribution();
  void initialize(unsigned int Mb_, unsigned int nnzbs_,  unsigned int sizeDvals_, unsigned int sizeDrows_, unsigned int sizeDcols_);
  void copyHostToDevice(std::vector<double> Dvals, std::vector<int> Drows, std::vector<int> Dcols, std::vector<double> z1);
  bool analyseMatrix();
  void ilu0Solver();
  std::vector<double> solveSytem();
};
