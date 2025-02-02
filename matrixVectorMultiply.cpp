/* classic */
void matrixVectorMultiply(const int* A, const int* u, int* v, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    { 
        int sum = 0; 
        for (int j = 0; j < cols; j++)
        { 
            sum += A[i * cols + j] * u[j]; 
        } 
        v[i] = sum; 
    } 
}

/* CUDA */
__global__ void matrixVectorMultiply(const int* A, const int* u, int* v, int rows, int cols)
{
    int size = rows * cols;
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int shared_u[cols];

    /* for each column do */
    for(int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        shared_u[i] = u[i];
    }
    __syncThreads();

    /* this works only in case threadsCount >= size */
    int sum = 0; 
    for (int j = 0; j < cols; j++)
    { 
        sum += A[threadId * cols + j] * shared_u[j]; 
    } 
    v[i] = sum; 
}
/* OMP */
void matrixVectorMultiply(const int* A, const int* u, int* v, int rows, int cols)
{
    unsigned size = rows*cols;
    #pragma acc enter data copyin(A[:size], u[:size]) copyout(v[:size])
    #pragma acc parallel loop
    for (int i = 0; i < rows; i++)
    {
        int sum = 0;
        #pragma acc seq
        for (int j = 0; j < cols; j++)
        { 
            sum += A[i * cols + j] * u[j]; 
        } 
        v[i] = sum; 
    } 
}