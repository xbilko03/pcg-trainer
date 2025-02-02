struct Velocities 
{ 
    int size = 0; 
    float3 *velocities;

    Velocities(int count) : size(count)
    { 
        velocities = new float3[count];
        #pragma acc enter data copyin(this) 
        #pragma acc enter data create(velocities[size]) 
    } 
    ~Velocities()
    { 
        #pragma acc exit data delete(velocities) 
        #pragma acc exit data delete(this) 
        delete[] velocities 
    } 
    void copy_to_gpu()
    { 
        #pragma acc update device(velocities[size]) 
    } 
    void copy_to_host()
    { 
        #pragma acc update host(velocities[size]) 
    } 
} 