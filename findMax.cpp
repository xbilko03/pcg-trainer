/* classic */
int findMax(const int* A, int size)
{ 
    int maxVal = A[0]; 
    for (int i = 1; i < size; i++)
    { 
        if (A[i] > maxVal)
        { 
            maxVal = A[i]; 
        } 
    } 
    return maxVal; 
}
/* CUDA */
__global__ int findMax(const int* A, int size, int* max)
{ 
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int __shared__ sharedMax[blockDim.x];

    int localMax = A[0];
    for (int i = threadId; i < size; i += gridDim.x * blockDim.x)
    {
        if (A[i] > localMax)
        {
            localMax = A[i];
        }
    }
    sharedMax[threadIdx.x] = localMax;
    __syncThreads();

    for (int stride = blockDimx.x / 2; stide > 0; stide = stide >> 1)
    {
        if(threadIdx.x < stride)
        {
            if(sharedMax[threadIdx.x] < sharedMax[threadIdx.x + stride])
            {
                sharedMax[threadIdx.x] = sharedMax[threadIdx.x + stride];
            }
        }
    }

    if(threadIdx.x)
    {
        AtomicMax(max, sharedMax[0]);
    }
}
/* OMP */
int findMax(const int* A, int size)
{ 
    int maxVal = A[0];
    #pragma acc enter data copyin(A[:size])
    #pragma acc parallel loop reduction(max:maxVal)
    for (int i = 1; i < size; i++)
    { 
        if (A[i] > maxVal)
        { 
            maxVal = A[i]; 
        } 
    } 
    return maxVal; 
}