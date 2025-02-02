/* classic */
void dotProduct(const int* A, const int* B, int size, int* result)
{
    result = 0; 
    for (int i = 0; i < size; i++)
    { 
        result += A[i] * B[i]; 
    } 
}

/* CUDA */
__global__ void dotProduct(const int* A, const int* B, int size, int* result)
{
    __shared__ int sharedResult[blockDim.x];
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId == 0)
    {
        result = 0;
        sharedResult = 0;
    }
    
    int localResult = 0;

    for (int i = threadId; i < size; i += blockDim.x)
    {
        localResult += A[i] * B[i];
    }

    sharedResult[threadIdx.x] = localResult;
    __syncThreads();
    
    for(int stride = blockDim.x / 2; i > 0; i = i >> 1)
    {
        if(threadIdx.x < stride)
        {
            sharedResult[threadIdx.x] += sharedResult[threadIdx.x + stride];
        }
    }
    if(threadIdx.x == 0)
    {
        atomicAdd(result, sharedResult[0]);
    }
}
/* OMP */
void dotProduct(const int* A, const int* B, int size, int* result)
{
    result = 0;
    #pragma acc data enter copyin(A[:size], B[:size], C[:size]) copyout(B[:size])
    #pragma acc parallel loop reduction(+:result)
    for (int i = 0; i < size; i++)
    { 
        result += A[i] * B[i]; 
    } 
}