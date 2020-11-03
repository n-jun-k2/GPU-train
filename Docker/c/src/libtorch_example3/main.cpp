/**
 * @brief 行列処理
 *  https://pytorch.org/cppdocs/notes/tensor_indexing.html
 *  https://pytorch.org/cppdocs/notes/tensor_creation.html
 *  https://pytorch.org/cppdocs/api/classat_1_1_tensor.html?highlight=tensor
 * 
 * https://pytorch.org/cppdocs/api/classc10_1_1_array_ref.html#exhale-class-classc10-1-1-array-ref
 * https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#exhale-class-classat-1-1-tensor
 * 
 * https://caffe2.ai/doxygen-c/html/classat_1_1_tensor.html
 * @date 2020/10/26
 * @author kawakami jun
 */

#include <torch/torch.h>

#include <iostream>

int main() {


    auto tensor_a = torch::tensor({1, 2, 3, 4, 5, 6});

    auto tensor_b = torch::tensor({1, 2, 3, 4, 5, 6});
    auto intarrayref = tensor_b.sizes(); // torch::IntArrayRef
    auto strides = tensor_b.strides(); // torch::IntArrayRef

    float data[] = { 1, 2, 3,
                    4, 5, 6 };
    auto tensor_blob = torch::from_blob(data, {2, 3});

    torch::IntArrayRef array_sample = {1, 2, 3, 4, 5};
    std::cout << "array_sample"         << array_sample                         << std::endl;
    std::cout << "a="                   << tensor_a                             << std::endl;
    std::cout << "a[1]="                << tensor_a[1]                          << std::endl;
    std::cout << "a[{2}]="              << tensor_a[{2}]                        << std::endl;
    std::cout << "slice_a="             << tensor_a.slice(0, 1, INT64_MAX, 2)   << std::endl;
    std::cout << "tensor_a.repeat ="    << tensor_a.repeat ({2, 3})             << std::endl;
    std::cout << "tensor_a.reshape="    << tensor_a.reshape({2, 3})             << std::endl;
    std::cout << "b="                   << tensor_b                             << std::endl;
    std::cout << "tensor_blob = "       << tensor_blob                          << std::endl;
    std::cout << "tensor_blob.t = "     << tensor_blob.t()                      << std::endl;
    std::cout << "tensor_blob.t_ = "    << tensor_blob.t_()                     << std::endl;
    std::cout << "tensor_blob.t = "     << tensor_blob.t()                      << std::endl;
    std::cout << "tensor_blob[0] = "    << tensor_blob[0]                       << std::endl;
    std::cout << "tensor_blob.size(0)=" << tensor_blob.size(0)                  << std::endl;
    std::cout << "tensor_blob.size(1)=" << tensor_blob.size(1)                  << std::endl;
    std::cout << "b.sizes="             << intarrayref                          << std::endl;
    std::cout << "b.strides="           << strides                              << std::endl;
    std::cout << "b.is_same="           << tensor_b.is_same(tensor_a)           << std::endl;
    std::cout << "b.dim="               << tensor_b.dim()                       << std::endl;
    std::cout << "b.numel="             << tensor_b.numel()                     << std::endl;
    std::cout << "b.defined="           << tensor_b.defined()                   << std::endl;
    std::cout << "b.nbytes="            << tensor_b.nbytes()                    << std::endl;
    std::cout << "b.dtype="             << tensor_b.dtype()                     << std::endl;
    std::cout << "b.layout="            << tensor_b.layout()                    << std::endl;
    std::cout << "b.device="            << tensor_b.device()                    << std::endl;
    std::cout << "b.options="           << tensor_b.options()                   << std::endl;
    std::cout << "b.get_device="        << tensor_b.get_device()                << std::endl;
    std::cout << "b.is_cuda="           << tensor_b.is_cuda()                   << std::endl;
    std::cout << "b.is_hip="            << tensor_b.is_hip()                    << std::endl;
    std::cout << "b.is_sparse="         << tensor_b.is_sparse()                 << std::endl;
    std::cout << "b.is_mkldnn="         << tensor_b.is_mkldnn()                 << std::endl;
    std::cout << "b.is_quantized="      << tensor_b.is_quantized()              << std::endl;
    std::cout << "b.itemsize="          << tensor_b.itemsize()                  << std::endl;
    std::cout << "b.is_alias_of="       << tensor_b.is_alias_of(tensor_a)       << std::endl;
    std::cout << "b.has_storage="       << tensor_b.has_storage()               << std::endl;
    std::cout << "b.element_size="      << tensor_b.element_size()              << std::endl;
    std::cout << "b.storage_offset="    << tensor_b.storage_offset()            << std::endl;
    std::cout << "b.toString="          << tensor_b.toString()                  << std::endl;
    std::cout << "b.ndimension ="       << tensor_b.ndimension ()               << std::endl;


}
