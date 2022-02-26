/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MseLossBackward : public Node {
 public:
  MseLossBackward(const Value& grad_output, const Value& input, const Value& target,
                  ReductionMode reduction);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  ReductionMode reduction() const {
    return reduction_;
  }

 private:
  ReductionMode reduction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors