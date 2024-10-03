#include "rocsolver-ilu.hpp"

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

#define ROCSOLVER_CALL(call)                                                   \
  do {                                                                         \
    rocblas_status err = call;                                               \
    if (rocblas_status_success != err) {                                     \
      printf("rocSPARSE ERROR (code = %d) at %s:%d\n", err, __FILE__,          \
             __LINE__);                                                        \
      assert(0);                                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

RocsolverMSWContribution::~RocsolverMSWContribution()
{
    ROCSOLVER_CALL(rocblas_destroy_handle(handle));
}

void RocsolverMSWContribution::initialize(int M, int N)
{
    this->M = M;
    this->N = N;
    this->lda = M > N ? M : N;
    this->ldb = M;

    int ipivDim = M > N ? N : M;
    HIP_CALL(hipMalloc(&ipiv, sizeof(rocblas_int)*ipivDim));
    HIP_CALL(hipMalloc(&d_Dmatrix_hip, sizeof(double)*M*N));
    HIP_CALL(hipMalloc(&z_hip, sizeof(double)*ldb*this->Nrhs));
    HIP_CALL(hipMalloc(&info, sizeof(rocblas_int)));
}

void RocsolverMSWContribution::copyHostToDevice(double* Dmatrix, std::vector<double> z1)
{
    HIP_CALL(hipMemcpy(d_Dmatrix_hip, Dmatrix, M*N*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(z_hip, z1.data(), ldb*Nrhs*sizeof(double), hipMemcpyHostToDevice));
}

std::vector<double> RocsolverMSWContribution::solveSytem()
{
    ROCSOLVER_CALL(rocblas_create_handle(&handle));

    ROCSOLVER_CALL(rocsolver_dgetrf(handle, M, N, d_Dmatrix_hip, lda, ipiv, info));

    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, N, Nrhs, d_Dmatrix_hip, lda, ipiv, z_hip, ldb));

    //HIP_CALL(hipDeviceSynchronize());

    vecSol.resize(M);
    HIP_CALL(hipMemcpy(vecSol.data(), z_hip, M*sizeof(double),hipMemcpyDeviceToHost));

    return vecSol;
}




