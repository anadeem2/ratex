/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/Operators.h>
#include <ATen/native/CPUFallback.h>

#include "ratex/csrc/aten_autograd_ops.h"

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "ratex/csrc/LazyNativeFunctions.h"
#include "ratex/csrc/aten_cpu_fallback.h"

namespace torch_lazy_tensors {
namespace aten_autograd_ops {

torch::Tensor MaxPool2dAutogradFunction::forward(torch::autograd::AutogradContext* ctx,
                                                 torch::Tensor self, torch::IntArrayRef kernel_size,
                                                 torch::IntArrayRef stride,
                                                 torch::IntArrayRef padding,
                                                 torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    auto results = FALLBACK_ATEN_OP(max_pool2d_with_indices, self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto outputs = LazyTensor::max_pool_nd(bridge::GetLtcTensor(self), /*spatial_dim_count=*/2,
                                         Helpers::I64List(kernel_size), Helpers::I64List(stride),
                                         Helpers::I64List(padding), ceil_mode);
  return bridge::AtenFromLtcTensor(std::get<0>(outputs));
}

torch::autograd::variable_list MaxPool2dAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  // Lowering when ceil_mode or dilation is set not supported yet.
  torch::Tensor grad;
  if (IsNonTrivialDilation(dilation)) {
    auto indices = saved[1];
    grad = FALLBACK_ATEN_OP(max_pool2d_with_indices_backward, grad_output[0], self, kernel_size,
                            stride, padding, dilation, ceil_mode, indices);
  }
  grad = bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::GetLtcTensor(grad_output[0]), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad, undef, undef, undef, undef, undef};
  return grad_inputs;
}

torch::Tensor MaxPool3dAutogradFunction::forward(torch::autograd::AutogradContext* ctx,
                                                 torch::Tensor self, torch::IntArrayRef kernel_size,
                                                 torch::IntArrayRef stride,
                                                 torch::IntArrayRef padding,
                                                 torch::IntArrayRef dilation, bool ceil_mode) {
  ctx->saved_data["kernel_size"] = kernel_size;
  ctx->saved_data["stride"] = stride;
  ctx->saved_data["padding"] = padding;
  ctx->saved_data["dilation"] = dilation;
  ctx->saved_data["ceil_mode"] = ceil_mode;
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    auto results = FALLBACK_ATEN_OP(max_pool3d_with_indices, self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
    ctx->save_for_backward({self, std::get<1>(results)});
    return std::get<0>(results);
  }
  ctx->save_for_backward({self});
  auto outputs = LazyTensor::max_pool_nd(bridge::GetLtcTensor(self), /*spatial_dim_count=*/3,
                                         Helpers::I64List(kernel_size), Helpers::I64List(stride),
                                         Helpers::I64List(padding), ceil_mode);
  return bridge::AtenFromLtcTensor(std::get<0>(outputs));
}

torch::autograd::variable_list MaxPool3dAutogradFunction::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
  auto kernel_size = ctx->saved_data["kernel_size"].toIntList().vec();
  auto stride = ctx->saved_data["stride"].toIntList().vec();
  auto padding = ctx->saved_data["padding"].toIntList().vec();
  auto dilation = ctx->saved_data["dilation"].toIntList().vec();
  auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
  auto saved = ctx->get_saved_variables();
  auto self = saved[0];
  // Lowering when ceil_mode or dilation is set not supported yet.
  torch::Tensor grad;
  if (IsNonTrivialDilation(dilation)) {
    auto indices = saved[1];
    grad = FALLBACK_ATEN_OP(max_pool3d_with_indices_backward, grad_output[0], self, kernel_size,
                            stride, padding, dilation, ceil_mode, indices);
  }
  grad = bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::GetLtcTensor(grad_output[0]), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode));

  torch::Tensor undef;
  torch::autograd::variable_list grad_inputs = {grad, undef, undef, undef, undef, undef};
  return grad_inputs;
}

torch::Tensor Dropout::forward(torch::autograd::AutogradContext* ctx,
                                                 const at::Tensor & input, double p, bool train) {

  ctx->saved_data["p"] = p;
  auto outputs = LazyTensor::dropout(bridge::GetLtcTensor(input), p, train);
  ctx->save_for_backward({bridge::AtenFromLtcTensor(std::get<1>(outputs)), bridge::AtenFromLtcTensor(std::get<2>(outputs))});
  return bridge::AtenFromLtcTensor(std::get<0>(outputs));
}

torch::autograd::variable_list Dropout::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {



  auto p = ctx->saved_data["p"];

  auto saved = ctx->get_saved_variables();
  auto mask = bridge::GetLtcTensor(saved[0]);
  auto reserved_space = bridge::GetLtcTensor(saved[1]);
  auto grad = bridge::GetLtcTensor(grad_output[0]);
  auto results = LazyTensor::dropout_backward(grad, mask, reserved_space);

  auto grad_outputs = bridge::AtenFromLtcTensor(results);
  torch::Tensor undef;
  torch::autograd::variable_list grad_result = {grad_outputs, undef, undef};

  return grad_result;
}

torch::Tensor MatMul::forward(torch::autograd::AutogradContext* ctx,
                                                 const at::Tensor & self, const at::Tensor & other) {


  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor other_tensor = bridge::GetLtcTensor(other);
  std::vector<int64_t> a_shape =
      lazy_tensors::util::ToVector<int64_t>(self_tensor.shape().get().dimensions());
  std::vector<int64_t> b_shape =
      lazy_tensors::util::ToVector<int64_t>(other_tensor.shape().get().dimensions());
  int64_t a_size = a_shape.size();
  int64_t b_size = b_shape.size();

  // return bridge::AtenFromLtcTensor(LazyTensor::matmul(self_tensor, other_tensor, a_shape, b_shape));


  // RATEX_VLOG(2) <<"a_shape=" << a_shape <<" b_shape=" << b_shape <<" a_size=" << a_size <<" b_size=" << b_size;

  int64_t unit_dim = -1;
  LazyTensor out;

  // TODO support higher demension conversion to lower dim matmul
  if (a_size > 4 || b_size > 4) {
    return FALLBACK_ATEN_OP(matmul, self, other);
  } else if (a_size > 2 && b_size > 2) {
    // 4 dim matrix w/ unit dim (m5 support)
    if (a_size > 3) {
      // Convert to 3D by squeezing unit dim, fallback if cannot squeeze
      unit_dim = std::find(a_shape.begin(), a_shape.end(), 1) - a_shape.begin();
      if (unit_dim == a_size)
        return FALLBACK_ATEN_OP(matmul, self, other);  // No unit dim - unsupported shape
      self_tensor = LazyTensor::squeeze(self_tensor, unit_dim);
    }
    if (b_size > 3) {
      // Convert to 3D by squeezing unit dim, fallback if cannot squeeze. Ensure if both a & b 4 dim
      // have unit dim on same axis
      auto b_dim = std::find(b_shape.begin(), b_shape.end(), 1) - b_shape.begin();
      bool missing_unit_dim = b_dim == b_size;
      bool broadcast = (b_dim == unit_dim || unit_dim == -1) ? false : true;
      if (missing_unit_dim || broadcast) return FALLBACK_ATEN_OP(matmul, self, other);
      unit_dim = b_dim;
      other_tensor = LazyTensor::squeeze(other_tensor, unit_dim);
    }

    out = LazyTensor::bmm(self_tensor, other_tensor);

    if (unit_dim != -1) out = LazyTensor::unsqueeze(out, unit_dim);

    // return aten_autograd_ops::MatMul::apply(bridge::AtenFromLtcTensor(out), bridge::AtenFromLtcTensor(self_tensor), bridge::AtenFromLtcTensor(other_tensor), a_size, b_size, unit_dim);

  } else {
    if (a_size > 2) {
    // a is 3D and b is < 3D, try to squeeze a
    unit_dim = std::find(a_shape.begin(), a_shape.end(), 1) - a_shape.begin();
    if (unit_dim == a_size)
      return FALLBACK_ATEN_OP(matmul, self, other);  // No unit dim - unsupported shape
    self_tensor = LazyTensor::squeeze(self_tensor, unit_dim);
  } else if (a_size == 1) {
    self_tensor = LazyTensor::unsqueeze(self_tensor, 0);
  }
  if (b_size > 2) {
    // b is 3D and a is < 3D, try to squeeze b
    unit_dim = std::find(b_shape.begin(), b_shape.end(), 1) - b_shape.begin();
    if (unit_dim == b_size)
      return FALLBACK_ATEN_OP(matmul, self, other);  // No unit dim - unsupported shape
    other_tensor = LazyTensor::squeeze(other_tensor, unit_dim);
  } else if (b_size == 1) {
    other_tensor = LazyTensor::unsqueeze(other_tensor, 1);
  }
  

  out = LazyTensor::matmul(self_tensor, other_tensor, "matmul");

  if (b_size == 1) out = LazyTensor::squeeze(out, 1);
  if (a_size == 1) out = LazyTensor::squeeze(out, 0);
  if (a_size > 2)
    out = LazyTensor::unsqueeze(out, 0);
  else if (b_size > 2)
    out = LazyTensor::unsqueeze(out, 1);

  }
  ctx->save_for_backward({bridge::AtenFromLtcTensor(self_tensor), bridge::AtenFromLtcTensor(other_tensor)});
  ctx->saved_data["a_size"] = a_size;
  ctx->saved_data["b_size"] = b_size;
  ctx->saved_data["a_shape"] = a_shape;
  ctx->saved_data["b_shape"] = b_shape;
  ctx->saved_data["unit_dim"] = unit_dim;


  //  return output;
  return bridge::AtenFromLtcTensor(out);
}

torch::autograd::variable_list MatMul::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
  
// TODO tranpose type handeling like RAF

  auto saved = ctx->get_saved_variables();
  int64_t a_size = ctx->saved_data["a_size"].toInt();;
  int64_t b_size = ctx->saved_data["b_size"].toInt();;
  int64_t unit_dim = ctx->saved_data["unit_dim"].toInt();;
  LazyTensor dy = bridge::GetLtcTensor(grad_output[0]);
    std::vector<int64_t> g_shape =
      lazy_tensors::util::ToVector<int64_t>(dy.shape().get().dimensions());
  int64_t g_size = g_shape.size();
  LazyTensor a = bridge::GetLtcTensor(saved[0]);
  LazyTensor b = bridge::GetLtcTensor(saved[1]);

      std::vector<int64_t> a_new =
      lazy_tensors::util::ToVector<int64_t>(a.shape().get().dimensions());

  LazyTensor grad_a;
  LazyTensor grad_b;

  auto a_shape = ctx->saved_data["a_shape"].toIntList().vec();
  auto b_shape = ctx->saved_data["b_shape"].toIntList().vec();

  std::cout<<" a_shape " << a_shape << " b_shape " << b_shape << " g_shape " << g_shape << "a_new "<< a_new;

  if(a_size > 2 && b_size > 2){
     if (g_size > 3) dy = LazyTensor::squeeze(dy);
  grad_b = LazyTensor::matmul(dy, b, "batch_matmul_nt");
  grad_a = LazyTensor::matmul(a, dy, "batch_matmul_tn");
  if (g_size > 3) {
    grad_b = LazyTensor::unsqueeze(grad_b, 0);
    grad_a = LazyTensor::unsqueeze(grad_a, 0);
  }
  else{
  if (a_size > 3) grad_a = LazyTensor::unsqueeze(grad_a, unit_dim);
  if (b_size > 3) grad_b = LazyTensor::unsqueeze(grad_b, unit_dim);
  }
  }
  else {
    if (g_size > 2) dy = LazyTensor::squeeze(dy);
  grad_b = LazyTensor::matmul(dy, b, "matmul_nt");
  grad_a = LazyTensor::matmul(a, dy, "matmul_tn");
  if (g_size > 2) {
    grad_b = LazyTensor::unsqueeze(grad_b, 0);
    grad_a = LazyTensor::unsqueeze(grad_a, 0);
  }
  else{
  if (b_size == 1) grad_b = LazyTensor::squeeze(grad_b); // 1
  else if (b_size > 2) grad_b = LazyTensor::unsqueeze(grad_b, 0); // dim = 1
  if (a_size == 1) grad_a = LazyTensor::squeeze(grad_a);
  else if (a_size > 2) grad_a = LazyTensor::unsqueeze(grad_a, 0);
  }
  }
  
  torch::autograd::variable_list grad_result = {bridge::AtenFromLtcTensor(grad_b), bridge::AtenFromLtcTensor(grad_a)};

  return grad_result;
}

}  // namespace aten_autograd_ops
}  // namespace torch_lazy_tensors
