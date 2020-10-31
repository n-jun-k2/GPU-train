/**
 * @file main.cpp
 * @author kawakami jun (you@domain.com)
 * @brief 
 * @date 2020-10-29
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <torch/torch.h>
#include <iostream>


int main() {

    /**
     * @brief CPUの行列に対してのデフォルトオプション
     */
    auto CPU_OPTIONS = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .layout(torch::kStrided)
                        .device(torch::kCPU)
                        .requires_grad(true);

    auto tensor_x = torch::tensor({1, 2, 3, 4}, CPU_OPTIONS);
    auto tensor_y = torch::rand(4, CPU_OPTIONS);
    auto tensor_A = torch::zeros({4, 4}, CPU_OPTIONS);

    /*
    補助定理(1):任意の列ベクトルyと0ではない列ベクトルxに対して、y=Axとなる行列Aが存在する。
    任意のj列目が(1 / x[j])*yであり、その他の列が0である行列 
    */
    const int j = 1;
    auto scalar_xj = tensor_y / tensor_x[j]; // (1 / x[j])* y
    for(auto&& i=0; i < tensor_A.size(0); ++i) {
        tensor_A[i][j] = scalar_xj[i];
    }

    std::cout << tensor_x << std::endl;
    std::cout << tensor_y << std::endl;
    std::cout << tensor_A << std::endl;
    auto tensor_xA = torch::matmul(tensor_A, tensor_x);
    std::cout << tensor_xA << std::endl;
    std::cout << "y == Ax :" << torch::all(tensor_y.eq(tensor_xA)) << std::endl;

    /*部分行列の作成*/
    auto tensor_B = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9}, CPU_OPTIONS).reshape({3, 3});
    auto tensor_Bdash = tensor_B.slice(0, 1).slice(1, 1);
    std::cout << "tensor_B" << std::endl;
    std::cout << tensor_B << std::endl;
    std::cout << "tensor_Bdash" << std::endl;
    std::cout << tensor_Bdash << std::endl;

    return 0;
}