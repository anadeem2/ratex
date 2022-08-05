/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class EmbeddingBackward : public Node {
 public:
  EmbeddingBackward(const Value& grad, const Value& indices, int64_t num_weights);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const int64_t num_weights() const{
    return num_weights_;
  }

  private:
    int64_t num_weights_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors