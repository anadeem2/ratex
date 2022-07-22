/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MatMul : public Node {
 public:
  MatMul(const Value& input, const Value& other, std::string type);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  std::string type() const {
    return type_;
  }

 private:
  std::string type_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
