/**
 * @brief ネットワークモデルの値、参照、ポインタ、move渡しのサンプル
 * @date 2020/10/25
 * @author kawakami jun
 */

#include <torch/torch.h>

#include <iostream>
#include <memory>

class Net : public torch::nn::Module { };

void func_a(Net net) {}
void func_b(Net& net) {}
void func_c(Net* net) {}
void func_d(std::weak_ptr<Net> net) {}

int main() {
	Net net;

	func_a(net);
	func_a(std::move(net));

	func_b(net);

	func_c(&net);

	auto sp_net = std::make_shared<Net>();
	func_d(sp_net);

}