#include <iostream>

int main()
{
    int devcount;
    cudaGetDeviceCount(&devcount);

    std::cout << "Dev Count: " << devcount << std::endl;

    cudaDeviceProp devProp;

    cudaGetDeviceProperties(&devProp, 0);

    std::string maxThreadsDim;
    std::string maxThreadsGrid;

    for (int i = 0; i < 3; i++)
    {
        maxThreadsDim = maxThreadsDim + " " + std::to_string(devProp.maxThreadsDim[i]);
    }

    for (int i = 0; i < 3; i++)
    {
        maxThreadsGrid = maxThreadsGrid + " " + std::to_string(devProp.maxGridSize[i]);
    }

    std::cout << "Max threads per dim: " << maxThreadsDim << std::endl;
    std::cout << "Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max number of registers: " << devProp.regsPerBlock << std::endl;
    std::cout << "Number of SMs: " << devProp.multiProcessorCount << std::endl;
    std::cout << "Max grid size: " << maxThreadsGrid << std::endl;

    return 0;
}