/* classic */
void matrixAdd(const int* A, const int* B, int* C, int rows, int cols)
{ 
    for (int i = 0; i < rows; i++)
    { 
        for (int j = 0; j < cols; j++)
        { 
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j]; 
        } 
    } 
}
/* CUDA */
__global__ void matrixAdd(const int* A, const int* B, int* C, int rows, int cols)
{ 
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = row; i < rows; i += blockDim.x * gridDim.x)
    {
        for (int j = 0; j < cols; j++)
        { 
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j]; 
        }
    }
}
/* OMP */
void matrixAdd(const int* A, const int* B, int* C, int rows, int cols)
{
    int size = rows * cols;
    #pragma acc enter data copyin(A[:size], B[:size]) copyout(C[:size])
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < rows; i++)
    { 
        for (int j = 0; j < cols; j++)
        { 
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j]; 
        } 
    } 
}