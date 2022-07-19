/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Unimplemented: public Node {
 public:
  Unimplemented(const Value& input, std::string name);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  std::string name() const {
    return name_;
  }
 private:
  // The dimension along which the result is computed.
  std::string name_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
