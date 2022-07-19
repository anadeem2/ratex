/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/ops/unimplemented.h"
#include "ratex/csrc/ops/raf_ops.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/util.h"

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Unimplemented::Unimplemented(const Value& input, std::string name)
    : Node(unimplemented, {input}, /*num_outputs=1*/ input.shape()), name_(name){
      // SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
    }

NodePtr Unimplemented::Clone(OpList operands) const {
  return MakeNode<Unimplemented>(operands.at(0), name_);
}

std::string Unimplemented::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()<< " name=" << name_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
