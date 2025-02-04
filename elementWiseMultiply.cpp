/* classic */
void elementWiseMultiply(const int* A, const int* B, int* C, int size)
{ 
    for (int i = 0; i < size; i++)
    { 
        C[i] = A[i] * B[i]; 
    }
}
/* CUDA */
__global__ void elementWiseMultiply(const int* A, const int* B, int* C, int size)
{
    threadId = threadIdx.x + blockIdx.x * blockDim.x;
    
    /* this works only in case threadsCount >= size */
    for(int i = threadId; i < size; i += blockDim.x * gridDim.x)
    {
        C[i] = A[i] * B[i];
    }
}
/* OMP */
void elementWiseMultiply(const int* A, const int* B, int* C, int size)
{
    #pragma acc enter data copyin(A[:size], B[:size]) copyout(C[:size])
    #pragma acc parallel loop
    for (int i = 0; i < size; i++)
    { 
        C[i] = A[i] * B[i]; 
    }
}