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

  /* 
  MultisegmentWellContribution()
  {
    rocsparse_create_handle(&handle);
  }

  ~MultisegmentWellContribution()
  {
    rocsparse_status status1 = rocsparse_destroy_handle(handle);
    if(status1 != rocsparse_status_success){
      std::cout <<"Could not destroy rocsparse handle" << std::endl;
    }

    HIP_CHECK(hipFree(temp_buffer));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_M));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_L));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_U));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

  }
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

  bool MultisegmentWellContribution::analyse_matrix()
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
    ROCSPARSE_CALL(rocsparse_dbsrilu0_buffer_size(handle, dir, Mb, nnzbs,
						  descr_M, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, dim_wells, ilu_info, &d_bufferSize_M));
    ROCSPARSE_CALL(rocsparse_dbsrsv_buffer_size(handle, dir, operation, Mb, nnzbs,
						descr_L, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, dim_wells, ilu_info, &d_bufferSize_L));
    ROCSPARSE_CALL(rocsparse_dbsrsv_buffer_size(handle, dir, operation, Mb, nnzbs,
						descr_U, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, dim_wells, ilu_info, &d_bufferSize_U));
    d_bufferSize = std::max(d_bufferSize_M, std::max(d_bufferSize_L, d_bufferSize_U));
    HIP_CALL(hipMalloc((void**)&d_buffer, d_bufferSize));

    // Perform analysis steps
    ROCSPARSE_CALL(rocsparse_dbsrilu0_analysis(handle, dir, \
                               Mb, nnzbs, descr_M, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, \
					       dim_wells, ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dbsrsv_analysis(handle, dir, operation, \
                             Mb, nnzbs, descr_L, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, \
					     dim_wells, ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dbsrsv_analysis(handle, dir, operation, \
                             Mb, nnzbs, descr_U, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, \
					     dim_wells, ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));

    // Check for zero pivot
    int zero_position = 0;
    rocsparse_status status = rocsparse_bsrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if (rocsparse_status_success != status) {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
        return false;
    }

    analysis_done = true;

    return true;
  }

  void MultisegmentWellContribution::ilu0_solver(double *z2, double *z1)
  {
    double one  = 1.0;

    if (analysis_done) {
       if (!analyse_matrix()) {
	 printf("Singular D matrix.");
       }
    }
    
    // Compute incomplete LU factorization
    ROCSPARSE_CALL(rocsparse_dbsrilu0(handle, dir, Mb, nnzbs, descr_M,
				      d_Dvals_hip, d_Drows_hip, d_Dcols_hip, dim_wells, ilu_info, rocsparse_solve_policy_auto, d_buffer));

    // Check for zero pivot
    int zero_position = 0;
    rocsparse_status status = rocsparse_bsrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if(rocsparse_status_success != status)
    {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
    }

    ROCSPARSE_CALL(rocsparse_dbsrsv_solve(handle, dir, \
                              operation, Mb, nnzbs, &one, \
					  descr_L, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, dim_wells, ilu_info, z1, z_aux, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CALL(rocsparse_dbsrsv_solve(handle, dir, \
                              operation, Mb, nnzbs, &one, \
					  descr_U, d_Dvals_hip, d_Drows_hip, d_Dcols_hip, dim_wells, ilu_info, z_aux, z2, rocsparse_solve_policy_auto, d_buffer));
    
  }

