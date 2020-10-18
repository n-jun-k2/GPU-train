#include <iostream>
#include <torch/torch.h>
int main()
{
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available!\n";
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "CPU only.\n";
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    auto tensor = torch::randn({2, 3});
    tensor = tensor.to(device);
    std::cout << tensor << std::endl;
}