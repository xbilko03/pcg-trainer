/* classic */
void columnWiseSum(const int* A, int* sums, int rows, int cols)
{ 
    for (int j = 0; j < cols; j++)
    { 
        int sum = 0; 
        for (int i = 0; i < rows; i++)
        { 
            sum += A[i * cols + j]; 
        } 
        sums[j] = sum; 
    } 
} 
/* CUDA */
__global__ void columnWiseSum(const int* A, int* sums, int rows, int cols)
{
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column >= cols) return;
    
    int sum = 0;
    for (int i = 0; i < rows; i++)
    { 
        sum += A[i * cols + column]; 
    } 
    sums[column] = sum; 
    
} 
/* OMP */
void columnWiseSum(const int* A, int* sums, int rows, int cols)
{
    int size = rows * cols;
    #pragma acc enter data copyin(A[:size]) copyout(sums[:size])
    #pragma acc parallel loop gang worker
    for (int j = 0; j < cols; j++)
    { 
        int sum = 0;
        #pragma acc loop vector reduction (+:sum)
        for (int i = 0; i < rows; i++)
        { 
            sum += A[i * cols + j]; 
        } 
        sums[j] = sum; 
    } 
} 
