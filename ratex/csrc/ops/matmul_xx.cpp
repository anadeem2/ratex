/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/ops/matmul_xx.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "ratex/csrc/ops/raf_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MatMulXX::MatMulXX(const Value& input, const Value& other, std::vector<int64_t> a_shape,
                   std::vector<int64_t> b_shape, std::string type)
    : Node(raf_matmul_xx, {input, other}, /*num_outputs=*/1,
           lazy_tensors::util::MHash(a_shape.size() + b_shape.size())),
      a_shape_(a_shape),
      b_shape_(b_shape),
      type_(type) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr MatMulXX::Clone(OpList operands) const {
  return MakeNode<MatMulXX>(operands.at(0), operands.at(1), a_shape_, b_shape_, type_);
}

std::string MatMulXX::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << " a_shape= " << a_shape_ << " b_shape= " << b_shape_
     << " type= " << type_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
