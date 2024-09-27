#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <rocsparse/rocsparse.h>

class MultisegmentWellContribution
{

private: 
  bool analysis_done = false;
  int nnzbs;
  rocsparse_int Mb;
  rocsparse_int *d_Drows_hip;
  rocsparse_int *d_Dcols_hip;
  double *d_Dvals_hip;
  void *d_buffer;
  rocsparse_handle handle;
  rocsparse_mat_descr descr_A, descr_M, descr_L, descr_U;
  rocsparse_mat_info ilu_info;
  rocsparse_direction dir = rocsparse_direction_row;
  rocsparse_operation operation = rocsparse_operation_none;
  double *z_aux;

public:
  MultisegmentWellContribution();
  ~MultisegmentWellContribution();
  bool analyse_matrix();
  void ilu0_solver(double *z2, double *z1);
  
  
};
