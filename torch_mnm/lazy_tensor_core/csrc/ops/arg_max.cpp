#include "lazy_tensor_core/csrc/ops/arg_max.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ArgMax::ArgMax(const Value& input, lazy_tensors::int64 dim, bool keepdim)
    : Node(ir::OpKind(at::aten::argmax), {input},
           /*num_outputs=*/1, lazy_tensors::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ArgMax::Clone(OpList operands) const {
  return MakeNode<ArgMax>(operands.at(0), dim_, keepdim_);
}

std::string ArgMax::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors