/* classic */
void scalarMultiply(const int* A, int scalar, int* B, int size)
{
    for (int i = 0; i < size; i++)
    { 
        B[i] = A[i] * scalar; 
    }
}

/* CUDA */
void scalarMultiply(const int* A, int scalar, int* B, int size)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    for(i = threadId; i < size; i += blockDim.x * gridDim.x)
    {
        B[i] = A[i] * scalar;
    }
}
/* OMP */
void scalarMultiply(const int* A, int scalar, int* B, int size)
{
    #pragma acc data enter copyin(A[:size]) copyout(B[:size])
    #pragma acc parallel loop
    for (int i = 0; i < size; i++)
    {
        B[i] = A[i] * scalar; 
    }
}