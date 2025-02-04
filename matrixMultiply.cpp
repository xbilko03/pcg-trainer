/* classic */
void matrixMultiply(const int* A, const int* B, int* C, int rowsA, int colsA, int colsB) 
{ 
    for (int i = 0; i < rowsA; i++)
    { 
        for (int j = 0; j < colsB; j++)
        { 
            int sum = 0; 
            for (int k = 0; k < colsA; k++)
            { 
                sum += A[i * colsA + k] * B[k * colsB + j]; 
            } 
            C[i * colsB + j] = sum; 
        } 
    } 
} 
/* CUDA */
__global__ void matrixMultiply(const int* A, const int* B, int* C, int rowsA, int colsA, int colsB) 
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row >= rowsA || col >= colsB) return;

    for (int i = row; i < rows; i += blockDim.y * gridDim.y)
    {
        for (int j = col; j < cols; j += blockDim.x * blockDim.y)
        {
            int sum = 0;
            for (int k = 0; k < colsA; k++)
            {
                sum += A[i * colsA + k] * B[k * colsB + col]; 
            }
            C[i * colsB + j] = sum;
        }
    }
}
/* OMP */
void matrixMultiply(const int* A, const int* B, int* C, int rowsA, int colsA, int colsB) 
{
    int sizeB = rowsA*colsA;
    int sizeA = rowsA*colsB;
    #pragma acc enter data copyin(A[:sizeA], B[:sizeB])
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < rowsA; i++)
    { 
        for (int j = 0; j < colsB; j++)
        { 
            int sum = 0;
            #pragma acc loop reduction(+:sum)
            for (int k = 0; k < colsA; k++)
            { 
                sum += A[i * colsA + k] * B[k * colsB + j]; 
            } 
            C[i * colsB + j] = sum; 
        } 
    } 
} 