/* classic */
void matrixTranspose(const int* A, int* T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            T[j * rows + i] = A[i * cols + j]; 
        }
    }
}
/* CUDA */
__global__ void matrixTranspose(const int* A, int* T, int rows, int cols)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (row >= rows || col >= cols) return;

    T[col * rows + row] = A[row * cols + col];
}
/* OMP */
void matrixTranspose(const int* A, int* T, int rows, int cols)
{
    int size = rows * cols;
    #pragma acc enter data copyin(A[:size]) copyout(T[:size])
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            T[j * rows + i] = A[i * cols + j]; 
        }
    }
}