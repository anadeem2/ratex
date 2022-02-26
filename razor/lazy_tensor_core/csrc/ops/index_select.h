/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class IndexSelect : public Node {
 public:
  IndexSelect(const Value& input, lazy_tensors::int64 dim, const Value& index);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 dim() const {
    return dim_;
  };

 private:
  lazy_tensors::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors