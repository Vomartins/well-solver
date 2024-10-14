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
      printf("rocSOLVER ERROR (code = %d) at %s:%d\n", err, __FILE__,          \
             __LINE__);                                                        \
      assert(0);                                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


__global__ void scalar_csr_kernel(const int m,
                                  const unsigned int *__restrict__ row_offsets,
                                  const unsigned int *__restrict__ cols,
                                  const double *__restrict__ vals,
                                  const double *__restrict__ x,
                                  double *__restrict__ y,
                                  const double alpha,
                                  const double beta)
{
  const int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row < m) {
    // determine the start and ends of each row
    int p = row_offsets[row];
    int q = row_offsets[row+1];

    // execute the full sparse row * vector dot product operation
    double sum = 0;
    for (int i = p; i < q; i++) {
      sum += vals[i] * x[cols[i]];
    }

    // write to memory
    if (beta == 0) {
      y[row] = alpha * sum;
    } else {
      y[row] = alpha * sum + beta * y[row];
    }
  }
}

__global__ void scalar_csc_kernel(const int n,                           // number of columns
                                  const unsigned int *__restrict__ col_offsets,  // column start offsets
                                  const unsigned int *__restrict__ rows,        // row indices of non-zero elements
                                  const double *__restrict__ vals,              // non-zero values
                                  const double *__restrict__ x,                 // input vector x
                                  double *__restrict__ y,                       // output vector y
                                  const double alpha,                           // scaling factor alpha
                                  const double beta)                            // scaling factor beta
{
    const int col = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure the thread works on a valid column
    if (col < n) {
        // Determine the start and end of each column
        int start = col_offsets[col];
        int end = col_offsets[col + 1];

        // Loop over the non-zero elements of the column
        for (int i = start; i < end; i++) {
            int row = rows[i];           // Get the row index for this element
            double value = vals[i];      // Get the value of this element

            // y[row] = alpha * (val * x[col]) + beta * y[row];
            atomicAdd(&y[row], alpha * value * x[col]);
        }
    }
}

// Kernel copied from hipKernels.cpp

template<class Scalar>
__global__ void spmv_kernel(const Scalar *vals,
                       const unsigned int *cols,
                       const unsigned int *rows,
                       const int N,
                       const Scalar *x,
                       Scalar *out)
{
    extern __shared__ Scalar tmp[];
    const unsigned int bsize = blockDim.x;
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idx_b = gid / bsize;
    const unsigned int idx_t = threadIdx.x;
    const unsigned int num_workgroups = gridDim.x;

    int row = idx_b;

    while (row < N) {
        int rowStart = rows[row];
        int rowEnd = rows[row+1];
        int rowLength = rowEnd - rowStart;
        Scalar local_sum = 0.0;
        for (int j = rowStart + idx_t; j < rowEnd; j += bsize) {
            int col = cols[j];
            local_sum += vals[j] * x[col];
        }

        tmp[idx_t] = local_sum;
        __syncthreads();

        int offset = bsize / 2;
        while(offset > 0) {
            if (idx_t < offset) {
                tmp[idx_t] += tmp[idx_t + offset];
            }
            __syncthreads();
            offset = offset / 2;
        }

        if (idx_t == 0) {
            out[row] = tmp[idx_t];
        }

        row += num_workgroups;
    }
}

template<class Scalar>
__global__ void residual_blocked_k(const Scalar *vals,
                                   const unsigned int *cols,
                                   const unsigned int *rows,
                                   const int Nb,
                                   const Scalar *x,
                                   const Scalar *rhs,
                                   Scalar *out,
                                   const unsigned int block_dimM,
                                   const unsigned int block_dimN)
{
    extern __shared__ Scalar tmp[];
    const unsigned int warpsize = warpSize;
    const unsigned int bsize = blockDim.x;
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idx_b = gid / bsize;
    const unsigned int idx_t = threadIdx.x;
    unsigned int idx = idx_b * bsize + idx_t;
    const unsigned int bsM = block_dimM;
    const unsigned int bsN = block_dimN;
    const unsigned int num_active_threads = (warpsize/bsM/bsN)*bsM*bsN;
    const unsigned int num_blocks_per_warp = warpsize/bsM/bsN;
    const unsigned int NUM_THREADS = gridDim.x;
    const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
    unsigned int target_block_row = idx / warpsize;
    const unsigned int lane = idx_t % warpsize;
    const unsigned int c = (lane / bsM) % bsN;
    const unsigned int r = lane % bsM;

    // for 3x3 blocks:
    // num_active_threads: 27 (CUDA) vs 63 (ROCM)
    // num_blocks_per_warp: 3 (CUDA) vs  7 (ROCM)
    int offsetTarget = warpsize == 64 ? 48 : 32;

    while(target_block_row < Nb){
        unsigned int first_block = rows[target_block_row];
        unsigned int last_block = rows[target_block_row+1];
        unsigned int block = first_block + lane / (bsM*bsN);
        Scalar local_out = 0.0;

        if(lane < num_active_threads){
            for(; block < last_block; block += num_blocks_per_warp){
                Scalar x_elem = x[cols[block]*bsN + c];
                Scalar A_elem = vals[block*bsM*bsN + c + r*bsN];
                local_out += x_elem * A_elem;
            }
        }

        // do reduction in shared mem
        tmp[lane] = local_out;

        for(unsigned int offset = block_dimM; offset <= offsetTarget; offset <<= 1)
        {
            if (lane + offset < warpsize)
            {
                tmp[lane] += tmp[lane + offset];
            }
            __syncthreads();
        }

        if(lane < bsM){
            unsigned int row = target_block_row*bsM + lane;
            out[row] = rhs[row] + tmp[lane];
        }
        target_block_row += num_warps_in_grid;
    }
}

void RocsolverMSWContribution::scalar_csr(int m, int threads_per_block, unsigned int* row_offsets, unsigned int* cols, double* vals, double* x, double* y, double alpha, double beta)
{
  int num_blocks = (m + threads_per_block - 1) / threads_per_block;
  dim3 grid(num_blocks, 1, 1);
  dim3 block(threads_per_block, 1, 1);
  scalar_csr_kernel<<<grid, block>>>(m, row_offsets, cols, vals, x, y, alpha, beta);

  HIP_CALL(hipGetLastError()); // Checks for any errors during kernel launch
  HIP_CALL(hipDeviceSynchronize()); // Synchronizes and ensures the kernel has finished
}

void RocsolverMSWContribution::scalar_csc(int n, int threads_per_block, unsigned int* col_offsets, unsigned int* rows, double* vals, double* x, double* y, double alpha, double beta)
{
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    scalar_csc_kernel<<<grid, block>>>(n, col_offsets, rows, vals, x, y, alpha, beta);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize to ensure completion
}

void RocsolverMSWContribution::spmv_k(int n, int threads_per_block, unsigned int* row_offsets, unsigned int* cols, double* vals, double* x, double* y)
{
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    spmv_kernel<<<grid, block>>>(vals, cols, row_offsets, n, x, y);

    HIP_CALL(hipGetLastError()); // Check for errors
    HIP_CALL(hipDeviceSynchronize()); // Synchronize to ensure completion
}

void RocsolverMSWContribution::residual_blocked(double* vals, unsigned int* cols, unsigned int* rows, double* x, double* rhs, double* out, int Nb, unsigned int block_dimM, unsigned int block_dimN)
{
  unsigned int blockDim = 32;
  unsigned int number_wg = std::ceil(Nb/blockDim);
  unsigned int num_work_groups = number_wg == 0 ? 1 : number_wg;
  std::cout << num_work_groups << std::endl;
  unsigned int gridDim = num_work_groups*blockDim;
  unsigned int shared_mem_size = blockDim*sizeof(double);

  residual_blocked_k<<<dim3(gridDim), dim3(blockDim), shared_mem_size>>>(vals, cols, rows, Nb, x, rhs, out, block_dimM, block_dimN);

  HIP_CALL(hipGetLastError()); // Check for errors
  HIP_CALL(hipDeviceSynchronize()); // Synchronize to ensure completion
}



RocsolverMSWContribution::~RocsolverMSWContribution()
{
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipFree(ipiv));
    HIP_CALL(hipFree(d_Dmatrix_hip));
    HIP_CALL(hipFree(d_Bvals_hip));
    HIP_CALL(hipFree(d_Bcols_hip));
    HIP_CALL(hipFree(d_Brows_hip));
    HIP_CALL(hipFree(x_hip));
    HIP_CALL(hipFree(z_hip));
    HIP_CALL(hipFree(rhs_hip));
    HIP_CALL(hipFree(info));
    ROCSOLVER_CALL(rocblas_destroy_handle(handle));
}

void RocsolverMSWContribution::initialize(rocblas_int M, rocblas_int N, int sizeBvals, int sizeBcols, int sizeBrows, int resSize)
{
    this->M = M;
    this->N = N;
    this->lda = M > N ? M : N;
    this->ldb = M;
    this->sizeBvals = sizeBvals;
    this->sizeBcols = sizeBcols;
    this->sizeBrows = sizeBrows;
    this->resSize = resSize;

    int ipivDim = M > N ? N : M;
    HIP_CALL(hipMalloc(&ipiv, sizeof(rocblas_int)*ipivDim));
    HIP_CALL(hipMalloc(&d_Dmatrix_hip, sizeof(double)*M*N));
    HIP_CALL(hipMalloc(&d_Bvals_hip, sizeof(double)*sizeBvals));
    HIP_CALL(hipMalloc(&d_Bcols_hip, sizeof(unsigned int)*sizeBcols));
    HIP_CALL(hipMalloc(&d_Brows_hip, sizeof(unsigned int)*sizeBrows));
    HIP_CALL(hipMalloc(&x_hip, sizeof(double)*resSize));
    HIP_CALL(hipMalloc(&z_hip, sizeof(double)*ldb*this->Nrhs));
    HIP_CALL(hipMalloc(&rhs_hip, sizeof(double)*ldb*this->Nrhs));
    HIP_CALL(hipMalloc(&info, sizeof(rocblas_int)));
}

void RocsolverMSWContribution::copyHostToDevice(double *Dmatrix,
                                                std::vector<double> Bvals,
                                                std::vector<unsigned int> Brows,
                                                std::vector<unsigned int> Bcols,
                                                std::vector<double> x)
{
    std::vector<double> rhs(ldb*Nrhs, 0.0);

    HIP_CALL(hipMemcpy(d_Dmatrix_hip, Dmatrix, M*N*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bvals_hip, Bvals.data(), sizeBvals*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Bcols_hip, Bcols.data(), sizeBcols*sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_Brows_hip, Brows.data(), sizeBrows*sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(x_hip, x.data(), resSize*sizeof(double), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(rhs_hip, rhs.data(), ldb*Nrhs*sizeof(double), hipMemcpyHostToDevice));
}

std::vector<double> RocsolverMSWContribution::solveSytem()
{

    //ROCSOLVER_CALL(rocblas_create_handle(&handle));

    ROCSOLVER_CALL(rocsolver_dgetrf(handle, M, N, d_Dmatrix_hip, lda, ipiv, info));

    rocblas_int info_host;
    HIP_CALL(hipMemcpy(&info_host, info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    if (info_host != 0) {
        std::cerr << "LU factorization failed at column " << info_host << std::endl;
        exit(1);  // or handle the error appropriately
    }

    ROCSOLVER_CALL(rocsolver_dgetrs(handle, operation, N, Nrhs, d_Dmatrix_hip, lda, ipiv, z_hip, ldb));

    //HIP_CALL(hipDeviceSynchronize());

    vecSol.resize(M);
    HIP_CALL(hipMemcpy(vecSol.data(), z_hip, M*sizeof(double),hipMemcpyDeviceToHost));

    return vecSol;
}

std::vector<double> RocsolverMSWContribution::apply()
{
  //scalar_csr(Bm, 32, d_Brows_hip, d_Bcols_hip, d_Bvals_hip, x_hip, z_hip, 1.0, 0.0);
  //scalar_csc(sizeBrows-1, 32, d_Brows_hip, d_Bcols_hip, d_Bvals_hip, x_hip, z_hip, 1.0, 0.0);
  //spmv_k(sizeBrows-1, 32, d_Brows_hip, d_Bcols_hip, d_Bvals_hip, x_hip, z_hip);

  residual_blocked(d_Bvals_hip, d_Bcols_hip, d_Brows_hip, x_hip, rhs_hip, z_hip, sizeBrows-1, 4, 3);

  z1.resize(ldb*Nrhs);
  HIP_CALL(hipMemcpy(z1.data(), z_hip, ldb*Nrhs*sizeof(double),hipMemcpyDeviceToHost));

  ROCSOLVER_CALL(rocblas_create_handle(&handle));

  //return z1;

  return solveSytem();
}









