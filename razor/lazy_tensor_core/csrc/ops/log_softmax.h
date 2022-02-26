/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// IR node for log(softmax) operation.
class LogSoftmax : public Node {
 public:
  LogSoftmax(const Value& input, lazy_tensors::int64 dim, c10::optional<at::ScalarType> dtype);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 dim() const {
    return dim_;
  }

  const c10::optional<at::ScalarType>& dtype() const {
    return dtype_;
  }

 private:
  // The dimension along which the result is computed.
  lazy_tensors::int64 dim_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors