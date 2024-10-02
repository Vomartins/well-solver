// opm includes

#include "rocsparse-ilu0.hpp"

/*
#define HIP_CHECK(STAT)                                  \
    do {                                                 \
        const hipError_t stat = (STAT);                  \
        if(stat != hipSuccess)                           \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseSolverBackend::hip ";       \
            oss << "error: " << hipGetErrorString(stat); \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)

#define ROCSPARSE_CHECK(STAT)                            \
    do {                                                 \
        const rocsparse_status stat = (STAT);            \
        if(stat != rocsparse_status_success)             \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseSolverBackend::rocsparse "; \
            oss << "error: " << stat;                    \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)
*/

#define ROCSPARSE_CALL(call)                                                   \
  do {                                                                         \
    rocsparse_status err = call;                                               \
    if (rocsparse_status_success != err) {                                     \
      printf("rocSPARSE ERROR (code = %d) at %s:%d\n", err, __FILE__,          \
             __LINE__);                                                        \
      assert(0);                                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define HIP_CALL(call)                                     \
  do {                                                     \
    hipError_t err = call;                                 \
    if (hipSuccess != err) {                               \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,  \
             hipGetErrorString(err), __FILE__, __LINE__);  \
      assert(0);                                           \
      exit(1);                                             \
    }                                                      \
  } while (0)

  MultisegmentWellContribution::~MultisegmentWellContribution() {
    HIP_CALL(hipFree(d_Dvals_hip));
    HIP_CALL(hipFree(d_Drows_hip));
    HIP_CALL(hipFree(d_Dcols_hip));
    HIP_CALL(hipFree(z_aux_hip));
    HIP_CALL(hipFree(z1_hip));
    HIP_CALL(hipFree(z2_hip));
    HIP_CALL(hipFree(d_buffer));
    ROCSPARSE_CALL(rocsparse_destroy_handle(handle));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_M));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_L));
    ROCSPARSE_CALL(rocsparse_destroy_mat_descr(descr_U));
    ROCSPARSE_CALL(rocsparse_destroy_mat_info(ilu_info));
  }

  void MultisegmentWellContribution::initialize(unsigned int M_, unsigned int nnzs_, unsigned int sizeDvals_, unsigned int sizeDrows_, unsigned int sizeDcols_)
  {
    this->M = M_;
    this->nnzs = nnzs_;
    this->sizeDvals = sizeDvals_;
    this->sizeDrows = sizeDrows_;
    this->sizeDcols = sizeDcols_;

    HIP_CALL(hipMalloc(&d_Dvals_hip, sizeof(double)*sizeDvals));
    HIP_CALL(hipMalloc(&d_Drows_hip, sizeof(rocsparse_int)*sizeDrows));
    HIP_CALL(hipMalloc(&d_Dcols_hip, sizeof(rocsparse_int)*sizeDcols));
    HIP_CALL(hipMalloc(&z_aux_hip, sizeof(double)*M));
    HIP_CALL(hipMalloc(&z1_hip, sizeof(double)*M));
    HIP_CALL(hipMalloc(&z2_hip, sizeof(double)*M));
  }

  void MultisegmentWellContribution::copyHostToDevice(std::vector<double> Dvals,  std::vector<int> Drows, std::vector<int> Dcols, std::vector<double> z1)
  {
    HIP_CALL(hipMemcpy(d_Dvals_hip, Dvals.data(), sizeDvals*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Drows_hip, Drows.data(), sizeDrows*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Dcols_hip, Dcols.data(), sizeDcols*sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(z1_hip, z1.data(), M*sizeof(double),hipMemcpyHostToDevice));

  }

  bool MultisegmentWellContribution::analyseMatrix()
  {
    std::size_t d_bufferSize_M, d_bufferSize_L, d_bufferSize_U, d_bufferSize;

    ROCSPARSE_CALL(rocsparse_create_handle(&handle));

    // Create matrix descriptor for matrices M, L, and U
    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_M));

    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_L));
    ROCSPARSE_CALL(rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
    ROCSPARSE_CALL(rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit));

    ROCSPARSE_CALL(rocsparse_create_mat_descr(&descr_U));
    ROCSPARSE_CALL(rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper));
    ROCSPARSE_CALL(rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit));

    // Create matrix info structure
    ROCSPARSE_CALL(rocsparse_create_mat_info(&ilu_info));
    // Obtain required buffer sizes
    ROCSPARSE_CALL(rocsparse_dcsrilu0_buffer_size(handle, M, nnzs,
						  descr_M, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, ilu_info, &d_bufferSize_M));
    ROCSPARSE_CALL(rocsparse_dcsrsv_buffer_size(handle, operation, M, nnzs,
						descr_L, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, ilu_info, &d_bufferSize_L));
    ROCSPARSE_CALL(rocsparse_dcsrsv_buffer_size(handle, operation, M, nnzs,
						descr_U, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, ilu_info, &d_bufferSize_U));
    d_bufferSize = std::max(d_bufferSize_M, std::max(d_bufferSize_L, d_bufferSize_U));
    HIP_CALL(hipMalloc(&d_buffer, d_bufferSize));

    // Perform analysis steps
    ROCSPARSE_CALL(rocsparse_dcsrilu0_analysis(handle, \
                               M, nnzs, descr_M, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, \
					        ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dcsrsv_analysis(handle, operation, \
                             M, nnzs, descr_L, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, \
					      ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dcsrsv_analysis(handle, operation, \
                             M, nnzs, descr_U, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, \
					     ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));

    // Check for zero pivot
    rocsparse_int zero_position;
    rocsparse_status status = rocsparse_csrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if (rocsparse_status_success != status) {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
        return false;
    }

    analysis_done = true;

    return true;
  }

  void MultisegmentWellContribution::ilu0Solver()
  {
    double one  = 1.0;

    if (!analysis_done) {
       if (!analyseMatrix()) {
          std::cerr << "Matrix analysis failed." << std::endl;
          return;
       }
    }

    // Compute incomplete LU factorization
    ROCSPARSE_CALL(rocsparse_dcsrilu0(handle, M, nnzs, descr_M,
				      d_Dvals_hip, d_Drows_hip, d_Dcols_hip, ilu_info, rocsparse_solve_policy_auto, d_buffer));

    // Check for zero pivot
    rocsparse_int zero_position;
    rocsparse_status status = rocsparse_csrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if(rocsparse_status_success != status)
    {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
    }

    ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle, \
                              operation, M, nnzs, &one, \
					  descr_L, d_Dvals_hip, d_Drows_hip, d_Dcols_hip,  ilu_info, z1_hip, z_aux_hip, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dcsrsv_solve(handle,\
                              operation, M, nnzs, &one, \
					  descr_U, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, ilu_info, z_aux_hip, z2_hip, rocsparse_solve_policy_auto, d_buffer));
  }

  std::vector<double> MultisegmentWellContribution::solveSytem()
  {

    ilu0Solver();

    vecSol.resize(M);
    HIP_CALL(hipMemcpy(vecSol.data(), z2_hip, M*sizeof(double),hipMemcpyDeviceToHost));

    return vecSol;
  }

