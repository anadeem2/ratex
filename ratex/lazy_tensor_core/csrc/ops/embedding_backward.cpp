/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/embedding_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

EmbeddingBackward::EmbeddingBackward(const Value& grad, const Value& indices, int64_t num_weights)
    : Node(ir::OpKind(at::aten::embedding_backward), {grad, indices}, 1, lazy_tensors::util::MHash(num_weights)), num_weights_(num_weights) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr EmbeddingBackward::Clone(OpList operands) const {
  return MakeNode<EmbeddingBackward>(operands.at(0), operands.at(1), num_weights_);
}

std::string EmbeddingBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << "num_weight= " << num_weights_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors