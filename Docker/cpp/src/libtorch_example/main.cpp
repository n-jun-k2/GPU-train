/**
 * @brief 単純なネットワークモデル作成(パラメタ登録、サブモジュール登録)
 * https://pytorch.org/cppdocs/api/namespace_torch.html
 * https://pytorch.org/cppdocs/api/namespace_torch__nn.html
 * https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#exhale-class-classtorch-1-1nn-1-1-module
 * @date 2020/10/25
 * @author kawakami jun
 */

#include <torch/torch.h>

#include <iostream>

/**
 * @brief 簡単なモデル定義
 */
class Net :public torch::nn::Module {

    torch::nn::Linear linear;
    torch::Tensor another_bias;

public:
    Net(int64_t N, int64_t M) //パラメタとサブモジュールをネットワークに登録
		: linear(register_module("linear", torch::nn::Linear(N, M))),
		another_bias(register_parameter("b", torch::randn(M))) 
		{}
    torch::Tensor forward(torch::Tensor input) {
        return linear(input) + another_bias;
    }
};

int main() {


    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;

	auto net = Net{4, 5};
	for (const auto& param : net.parameters()){
		std::cout << param << std::endl;
	}

	for (const auto& param_pair : net.named_parameters()) {
		std::cout << param_pair.key() << ":" << param_pair.value() << std::endl;
	}

	// forward実行
	std::cout << net.forward(torch::ones({2, 4})) << std::endl;


}