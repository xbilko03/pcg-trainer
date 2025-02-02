/* classic */
void rowWiseSum(const int* A, int* sums, int rows, int cols)
{ 
    for (int i = 0; i < rows; i++)
    { 
        int sum = 0; 
        for (int j = 0; j < cols; j++)
        { 
            sum += A[i * cols + j]; 
        } 
        sums[i] = sum; 
    }
}
/* CUDA */
__global__ void rowWiseSum(const int* A, int* sums, int rows, int cols)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows * cols) return;

    int sum = 0;
    for (int j = 0; j < cols; j++)
    { 
        sum += A[row * cols + j]; 
    } 
    sums[row] = sum;
}
/* OMP */
void rowWiseSum(const int* A, int* sums, int rows, int cols)
{ 
    int size = cols * rows;
    #pragma acc enter data copyin(A[:size], rows, cols) copyout(sums[:size])
    #pragma acc parallel loop gang worker
    for (int i = 0; i < rows; i++)
    { 
        int sum = 0;
        #pragma acc loop vector reduction(+:sum)
        for (int j = 0; j < cols; j++)
        { 
            sum += A[i * cols + j]; 
        } 
        sums[i] = sum; 
    }
}