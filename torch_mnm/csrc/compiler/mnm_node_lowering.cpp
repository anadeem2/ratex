#include "./mnm_node_lowering.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool2d.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool3d.h"
#include "lazy_tensor_core/csrc/ops/all.h"
#include "lazy_tensor_core/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"
#include "lazy_tensor_core/csrc/ops/amp_update_scale.h"
#include "lazy_tensor_core/csrc/ops/any.h"
#include "lazy_tensor_core/csrc/ops/arg_max.h"
#include "lazy_tensor_core/csrc/ops/arg_min.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy_backward.h"
#include "lazy_tensor_core/csrc/ops/bitwise_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/cholesky.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/cumprod.h"
#include "lazy_tensor_core/csrc/ops/cumsum.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/diagonal.h"
#include "lazy_tensor_core/csrc/ops/diagonal_view_update.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/flip.h"
#include "lazy_tensor_core/csrc/ops/gather.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"
#include "lazy_tensor_core/csrc/ops/hardshrink.h"
#include "lazy_tensor_core/csrc/ops/hardtanh_backward.h"
#include "lazy_tensor_core/csrc/ops/index_along_dim.h"
#include "lazy_tensor_core/csrc/ops/index_get.h"
#include "lazy_tensor_core/csrc/ops/index_put.h"
#include "lazy_tensor_core/csrc/ops/index_select.h"
#include "lazy_tensor_core/csrc/ops/kth_value.h"
#include "lazy_tensor_core/csrc/ops/l1_loss.h"
#include "lazy_tensor_core/csrc/ops/l1_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"
#include "lazy_tensor_core/csrc/ops/linear_interpolation.h"
#include "lazy_tensor_core/csrc/ops/log_base.h"
#include "lazy_tensor_core/csrc/ops/log_softmax.h"
#include "lazy_tensor_core/csrc/ops/log_softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/masked_fill.h"
#include "lazy_tensor_core/csrc/ops/masked_scatter.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/max_unpool_nd.h"
#include "lazy_tensor_core/csrc/ops/max_unpool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/mean.h"
#include "lazy_tensor_core/csrc/ops/mse_loss.h"
#include "lazy_tensor_core/csrc/ops/mse_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d_backward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/normal.h"
#include "lazy_tensor_core/csrc/ops/not_supported.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/prod.h"
#include "lazy_tensor_core/csrc/ops/put.h"
#include "lazy_tensor_core/csrc/ops/qr.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d_backward.h"
#include "lazy_tensor_core/csrc/ops/replication_pad.h"
#include "lazy_tensor_core/csrc/ops/replication_pad_backward.h"
#include "lazy_tensor_core/csrc/ops/resize.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise_backward.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/scatter.h"
#include "lazy_tensor_core/csrc/ops/scatter_add.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/shrink_backward.h"
#include "lazy_tensor_core/csrc/ops/softmax.h"
#include "lazy_tensor_core/csrc/ops/softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/softshrink.h"
#include "lazy_tensor_core/csrc/ops/split.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/std.h"
#include "lazy_tensor_core/csrc/ops/sum.h"
#include "lazy_tensor_core/csrc/ops/svd.h"
#include "lazy_tensor_core/csrc/ops/symeig.h"
#include "lazy_tensor_core/csrc/ops/threshold.h"
#include "lazy_tensor_core/csrc/ops/threshold_backward.h"
#include "lazy_tensor_core/csrc/ops/topk.h"
#include "lazy_tensor_core/csrc/ops/triangular_solve.h"
#include "lazy_tensor_core/csrc/ops/tril.h"
#include "lazy_tensor_core/csrc/ops/triu.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d_backward.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d_backward.h"
#include "lazy_tensor_core/csrc/ops/var.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/shape_util.h"

#include "torch_mnm/csrc/ops/relay_expr.h"
#include "torch_mnm/csrc/ops/relay_function.h"
#include "torch_mnm/csrc/ops/log_softmax_backward_use_in.h"
#include "torch_mnm/csrc/ops/mnm_ops.h"

#include "./mnm_lowering_context.h"
#include "./mnm_shape_infer.h"
#include "./utils.h"

#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/value.h"
#include "mnm/binding.h"
#include "mnm/pass.h"
#include "meta/src/op/regs/schema2value.h"
#include "meta/src/common/shape_utils.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace {

using namespace mnm_backend;
using namespace mnm;
using namespace mnm::tensor;
using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::binding;
using namespace mnm::pass;
using mnm::pass::extract_binding::ExtractBinding;
using mnm::op::regs::schema2value::TupleInt;
using mnm::op::regs::schema2value::Int;
using mnm::op::regs::schema2value::Bool;
using mnm::op::regs::schema2value::String;

#define DECLARE_OP(name) Var Lower##name(const ir::Node* node)
#define DECLARE_OP2(name) \
  Var Lower##name(const ir::ops::name* node)

class MNMNodeLowering : public NodeLowering {
 public:
  MNMNodeLowering(ir::LoweringContext* loctx) : NodeLowering(loctx) {}

  bool Lower(const ir::Node* node) override;

  lazy_tensors::Shape Infer(const ir::Node* node) override;

  mnm_backend::MNMLoweringContext* loctx() {
    return static_cast<mnm_backend::MNMLoweringContext*>(loctx_);
  }

  Var LowerToMNM(const ir::Node* node);

 private:
  std::tuple<Var, Var> BinaryOpMatchTypes(const ir::Output& a, const ir::Output& b);

  Var LowerBitwise(const ir::Node* node);
  Var LowerAdd(const ir::Node* node);
  Var LowerDiv(const ir::Node* node);
  Var LowerMul(const ir::Node* node);
  Var LowerDeviceData(const ir::ops::DeviceData* node);
  // Var LowerAsStridedViewUpdate(
  //     const ir::ops::AsStridedViewUpdate* node);
  // Var LowerAsStrided(const ir::ops::AsStrided* node);
  Var LowerExpand(const ir::ops::Expand* node);
  Var LowerNotSupported(const ir::ops::NotSupported* node);
  template <class NllLossType>
  Var LowerNllLoss(const NllLossType* node);
  template <class NllLossBackwardType>
  Var LowerNllLossBackward(const NllLossBackwardType* node);
  DECLARE_OP(Ne);
  DECLARE_OP(Eq);
  DECLARE_OP(Gt);
  DECLARE_OP(Ceil);
  DECLARE_OP(Abs);
  DECLARE_OP2(Constant);
  DECLARE_OP2(Sum);
  DECLARE_OP2(Scalar);
  DECLARE_OP(Relu);
  DECLARE_OP(Neg);
  DECLARE_OP2(Permute);
  DECLARE_OP2(MaxPoolNdBackward);
  // DECLARE_OP2(ConvolutionBackwardOverrideable);
  DECLARE_OP(Mm);
  // DECLARE_OP2(View);
  DECLARE_OP(AddMatMul);
  DECLARE_OP2(ThresholdBackward);
  // DECLARE_OP2(NativeBatchNormForward);
  // DECLARE_OP2(NativeBatchNormBackward);
  // DECLARE_OP2(Mean);
  // DECLARE_OP2(LogSoftmaxBackward);
  DECLARE_OP2(MaxPoolNd);
  DECLARE_OP2(LogSoftmax);
  DECLARE_OP2(ConvolutionOverrideable);
  DECLARE_OP2(AdaptiveAvgPool2d);
  DECLARE_OP2(GenericSlice);
  DECLARE_OP2(View);
  DECLARE_OP2(AsStridedViewUpdate);
  DECLARE_OP2(AsStrided);
  DECLARE_OP2(Cast);
  DECLARE_OP2(LogSoftmaxBackwardUseIn);
  DECLARE_OP2(RelayExpr);
  DECLARE_OP2(RelayFunction);
  // DECLARE_OP2(TupleGetItem);
  // DECLARE_OP2(Tuple);
  lazy_tensors::Shape InferNe(const ir::Node* node);
  lazy_tensors::Shape InferEq(const ir::Node* node);
  lazy_tensors::Shape InferGt(const ir::Node* node);
  lazy_tensors::Shape InferExpand(const ir::ops::Expand* node);
  lazy_tensors::Shape InferBitwise(const ir::Node* node);
  lazy_tensors::Shape InferNllLoss(const ir::ops::NllLoss* node);
  lazy_tensors::Shape InferNllLossBackward(const ir::ops::NllLossBackward* node);
  lazy_tensors::Shape InferRelayExpr(const ir::ops::RelayExpr* node);
  lazy_tensors::Shape InferRelayFunction(const ir::ops::RelayFunction* node);
  // lazy_tensors::Shape InferTupleGetItem(const ir::ops::TupleGetItem* node);
  // lazy_tensors::Shape InferTuple(const ir::ops::Tuple* node);
  lazy_tensors::Shape InferAsStridedViewUpdate(const ir::ops::AsStridedViewUpdate* node);
  lazy_tensors::Shape InferCast(const ir::ops::Cast* node);
  lazy_tensors::Shape InferSum(const ir::ops::Sum* node);
};

#undef DECLARE_OP2
#undef DECLARE_OP

bool MNMNodeLowering::Lower(const ir::Node* node) {
  Var ops = LowerToMNM(node);
  if (node->num_outputs() > 1) {
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      loctx()->AssignOutputOp(ir::Output(node, i), BindSymbol(TupleGetItem(ops, i)));
    }
  } else {
    loctx()->AssignOutputOp(ir::Output(node, 0), ops);
  }
  return true;
}

#define HANDLE_GENERIC_OP(name, sym) \
  case sym: {                        \
    return Lower##name(node);        \
  }

#define HANDLE_GENERIC_OP2(name, sym)                                       \
  case sym: {                                                               \
    return Lower##name(ir::NodeCast<ir::ops::name>(node, ir::OpKind(sym))); \
  }

Var MNMNodeLowering::LowerToMNM(const ir::Node* node) {
  switch (node->op().op) {
    HANDLE_GENERIC_OP(Add, at::aten::add)
    HANDLE_GENERIC_OP(Div, at::aten::div)
    HANDLE_GENERIC_OP(Mul, at::aten::mul)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__and__)
    HANDLE_GENERIC_OP(Relu, at::aten::relu)
    HANDLE_GENERIC_OP(Ceil, at::aten::ceil)
    HANDLE_GENERIC_OP(Neg, at::aten::neg)
    HANDLE_GENERIC_OP(Ne, at::aten::ne)
    HANDLE_GENERIC_OP(Eq, at::aten::eq)
    HANDLE_GENERIC_OP(Gt, at::aten::gt)
    HANDLE_GENERIC_OP(Abs, at::aten::abs)
    HANDLE_GENERIC_OP2(Permute, at::aten::permute)
    HANDLE_GENERIC_OP2(MaxPoolNdBackward,
                       at::aten::max_pool2d_with_indices_backward)
    // HANDLE_GENERIC_OP2(ConvolutionBackwardOverrideable,
    //                    at::aten::convolution_backward_overrideable)
    HANDLE_GENERIC_OP(Mm, at::aten::mm)
    // HANDLE_GENERIC_OP2(View, at::aten::view)
    HANDLE_GENERIC_OP2(NllLoss, at::aten::nll_loss)
    HANDLE_GENERIC_OP2(NllLossBackward, at::aten::nll_loss_backward)
    HANDLE_GENERIC_OP2(Expand, at::aten::expand)
    HANDLE_GENERIC_OP(AddMatMul, at::aten::addmm)
    HANDLE_GENERIC_OP2(ThresholdBackward, at::aten::threshold_backward)
    // HANDLE_GENERIC_OP2(NativeBatchNormForward, at::aten::native_batch_norm)
    // HANDLE_GENERIC_OP2(NativeBatchNormBackward,
    //                    at::aten::native_batch_norm_backward)
    // HANDLE_GENERIC_OP2(Mean, at::aten::mean)
    // HANDLE_GENERIC_OP2(LogSoftmaxBackward, at::aten::_log_softmax_backward_data)
    HANDLE_GENERIC_OP2(MaxPoolNd, at::aten::max_pool2d)
    HANDLE_GENERIC_OP2(LogSoftmax, at::aten::log_softmax)
    HANDLE_GENERIC_OP2(ConvolutionOverrideable,
                       at::aten::convolution_overrideable)
    HANDLE_GENERIC_OP2(View, at::aten::view)
    HANDLE_GENERIC_OP2(AsStrided, at::aten::as_strided)
    HANDLE_GENERIC_OP2(Sum, at::aten::sum)
    case at::prim::Constant: {
      // TODO(asuhan): rework to remove ambiguity between Scalar and Constant
      // nodes to make dynamic_cast unnecessary.
      auto scalar_node = dynamic_cast<const ir::ops::Scalar*>(node);
      if (scalar_node) {
        return LowerScalar(scalar_node);
      }
      auto constant_node = dynamic_cast<const ir::ops::Constant*>(node);
      LTC_CHECK(constant_node);
      return LowerConstant(constant_node);
    }
    default: {
      if (node->op() == *ir::ops::ltc_cast) {
        return LowerCast(ir::NodeCast<ir::ops::Cast>(node, *ir::ops::ltc_cast));
      }
      if (node->op() == *ir::ops::ltc_device_data) {
        return LowerDeviceData(
            ir::NodeCast<ir::ops::DeviceData>(node, *ir::ops::ltc_device_data));
      }
      if (node->op() == *ir::ops::ltc_generic_slice) {
        return LowerGenericSlice(ir::NodeCast<ir::ops::GenericSlice>(
            node, *ir::ops::ltc_generic_slice));
      }
      if (node->op() == *ir::ops::ltc_as_strided_view_update) {
        return LowerAsStridedViewUpdate(
            ir::NodeCast<ir::ops::AsStridedViewUpdate>(
                node, *ir::ops::ltc_as_strided_view_update));
      }
      if (node->op() == *ir::ops::mnm_relay_expr) {
        return LowerRelayExpr(
            ir::NodeCast<ir::ops::RelayExpr>(
                node, *ir::ops::mnm_relay_expr));
      }
      if (node->op() == *ir::ops::mnm_relay_function) {
        return LowerRelayFunction(
            ir::NodeCast<ir::ops::RelayFunction>(
                node, *ir::ops::mnm_relay_function));
      }
      if (node->op() == *ir::ops::mnm_log_softmax_backward_use_in) {
        return LowerLogSoftmaxBackwardUseIn(
            ir::NodeCast<ir::ops::LogSoftmaxBackwardUseIn>(
                node, *ir::ops::mnm_log_softmax_backward_use_in));
      }
      // if (node->op() == *ir::ops::ltc_as_strided_view_update) {
      //   return LowerAsStridedViewUpdate(
      //       ir::NodeCast<ir::ops::AsStridedViewUpdate>(
      //           node, *ir::ops::ltc_as_strided_view_update));
      // }
    }
  }
  LTC_LOG(FATAL) << "NotImplementedError: " << *node;
  return {};
}

#undef HANDLE_GENERIC_OP2
#undef HANDLE_GENERIC_OP

std::tuple<Var, Var> MNMNodeLowering::BinaryOpMatchTypes(const ir::Output& a, const ir::Output& b) {
  using tvm::runtime::DLDataType2String;
  Var op0 = loctx()->GetOutputOp(a), op1 = loctx()->GetOutputOp(b);
  DType dtype_a = ToMNMDType(a.shape().element_type()), dtype_b = ToMNMDType(b.shape().element_type());
  LTC_CHECK_EQ(dtype_a.code, dtype_b.code);
  LTC_CHECK_EQ(dtype_a.lanes, dtype_b.lanes);
  if (dtype_a.bits < dtype_b.bits) {
    return std::make_tuple(BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast"), {op0,
    MakeConstant(String(DLDataType2String(dtype_b)))})), op1);
  } else if (dtype_a.bits > dtype_b.bits) {
    return std::make_tuple(op0, BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast"), {op1,
    MakeConstant(String(DLDataType2String(dtype_a)))})));
  } else {
    return std::make_tuple(op0, op1);
  }
}

ir::Output SimplifyBinaryInputs(const ir::Output& x, const ir::Output& y) {
  // lazy_tensors::Shape shape = node->operand(0).node->shape();
  // node->op().op
  if (x.node->op().op == at::aten::expand) {
    lazy_tensors::Shape x_shape = x.shape();
    lazy_tensors::Shape y_shape = y.shape();
    lazy_tensors::Shape in_shape = x.node->operand(0).shape();
    lazy_tensors::Shape in_y_shape = Helpers::GetPromotedBinaryOpShape(in_shape, y_shape);
    lazy_tensors::Shape x_y_shape = Helpers::GetPromotedBinaryOpShape(x_shape, y_shape);
    if (x_y_shape == in_y_shape) {
      return x.node->operand(0);
    }
  }
  return x;
}

Var MNMNodeLowering::LowerAdd(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.add"), {op0, op1, MakeNull(), MakeNull()}));
}

Var MNMNodeLowering::LowerDiv(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.divide"), {op0, op1}));
}

Var MNMNodeLowering::LowerMul(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var op0, op1;
  ir::Output output0 = SimplifyBinaryInputs(node->operand(0), node->operand(1));
  ir::Output output1 = SimplifyBinaryInputs(node->operand(1), output0);
  std::tie(op0, op1) = BinaryOpMatchTypes(output0, output1);
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.multiply"), {op0, op1}));
}

Var BuildBitwise(const std::vector<Var>& ops, const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  switch (node->op().op) {
    case at::aten::__and__: {
      return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.logical_and"), {ops[0], ops[1]}));
    }
    // case at::aten::__or__: {
    //   return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.or"), {ops[0], ops[1]}));
    // }
    // case at::aten::__xor__: {
    //   return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.xor"), {ops[0], ops[1]}));
    // }
  }
  LTC_LOG(FATAL) << "Invalid bitwise operator: " << node->op();
}

Var MNMNodeLowering::LowerBitwise(const ir::Node* node) {
  Var op0 = loctx()->GetOutputOp(node->operand(0));
  Var op1 = loctx()->GetOutputOp(node->operand(1));
  return BuildBitwise({op0, op1}, node);
}

Var MNMNodeLowering::LowerDeviceData(const ir::ops::DeviceData* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  return loctx()->GetParameter(node->data());
}

Var MNMNodeLowering::LowerLogSoftmax(const ir::ops::LogSoftmax* node) {
  // TODO(@hzfan): what is the dtype used for?
  Var input = loctx()->GetOutputOp(node->operand(0));
  Expr dim = MakeConstant(ScalarValue::make((int64_t)node->dim()));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.log_softmax"), {input, dim}));
}

Var MNMNodeLowering::LowerMaxPoolNd(const ir::ops::MaxPoolNd* node) {
  // TODO(@hzfan): return {result, indices}
  Var input = loctx()->GetOutputOp(node->operand(0));
  Expr kernel = MakeConstant(TupleInt(node->kernel_size()));
  Expr stride = MakeConstant(TupleInt(node->stride()));
  Expr padding = MakeConstant(TupleInt(node->padding()));
  Expr dilation = MakeConstant(TupleInt({1}));
  Expr ceil_mode = MakeConstant(Bool(node->ceil_mode()));
  Expr include_pad = MakeConstant(Bool(true));
  Expr layout = MakeConstant(String("NCHW"));
  Var result = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.max_pool2d"),
    {input, kernel, stride, padding, dilation, ceil_mode, include_pad, layout}));
  Var ret = BindSymbol(mnm::ir::Tuple(Array<Expr>({result, mnm::ir::Tuple(Array<Expr>({}))})));
  return ret;
}

Var MNMNodeLowering::LowerMaxPoolNdBackward(
    const ir::ops::MaxPoolNdBackward* node) {
  // TODO(@hzfan): max_pool2d_dx needs y
  Var grad_output = loctx()->GetOutputOp(node->operand(0));
  Var input = loctx()->GetOutputOp(node->operand(1));
  Expr kernel = MakeConstant(TupleInt(node->kernel_size()));
  Expr stride = MakeConstant(TupleInt(node->stride()));
  Expr padding = MakeConstant(TupleInt(node->padding()));
  Expr dilation = MakeConstant(TupleInt({1}));
  Expr ceil_mode = MakeConstant(Bool(node->ceil_mode()));
  Expr include_pad = MakeConstant(Bool(true));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.max_pool2d_dx"),
    {input, grad_output, kernel, stride, padding, dilation, ceil_mode, include_pad}));
}

// Var MNMNodeLowering::LowerMaxUnary(const ir::Node* node) {
//   Var x = loctx()->GetOutputOp(node->operand(0));
//   return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.max"), {x}));
// }

// Var MNMNodeLowering::LowerMax(const ir::Node* node) {
//   LTC_CHECK_EQ(node->num_outputs(), 1);
//   Var input0 = loctx()->GetOutputOp(node->operand(0));
//   Var input1 = loctx()->GetOutputOp(node->operand(1));
//   return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.maximum"), {input0, input1}));
// }

Var MNMNodeLowering::LowerRelu(const ir::Node* node) {
    LTC_CHECK_EQ(node->num_outputs(), 1);
    Var x = loctx()->GetOutputOp(node->operand(0));
    return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.relu"), {x}));
}

Var BuildSum(const std::vector<Var>& ops, const ir::ops::Sum* node) {
  LTC_CHECK_EQ(ops.size(), 1U);
  Var x = ops[0];
  std::vector<int64_t> dimension_0 = node->dimensions();
  std::vector<int64_t> keep_reduced_dimension_0(dimension_0.size(),
    static_cast<int64_t>(node->keep_reduced_dimensions()));
  Expr dimension = MakeConstant(TupleInt(dimension_0));
  Expr keep_reduced_dimension = MakeConstant(TupleInt(keep_reduced_dimension_0));
  Expr exclude = MakeConstant(Bool(false));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.sum"),
    {x, dimension, keep_reduced_dimension, exclude}));
}

Var MNMNodeLowering::LowerSum(const ir::ops::Sum* node) {
  // TODO(@hzfan): handle dtype
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildSum({x}, node);
}

template <class NllLossType>
Var BuildNllLoss(const std::vector<Var>& ops, const NllLossType* node) {
  // TODO(@hzfan): handle weight, reduction and ignore_index
  LTC_CHECK_EQ(ops.size(), 2U);
  Var logits = ops[0];
  Var labels = ops[1];
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.nll_loss"),
    {labels, logits}));
}


template <class NllLossType>
Var MNMNodeLowering::LowerNllLoss(const NllLossType* node) {
  Var logits = loctx()->GetOutputOp(node->operand(0));
  Var labels = loctx()->GetOutputOp(node->operand(1));
  return BuildNllLoss({logits, labels}, node);
}

template <class NllLossBackwardType>
Var BuildNllLossBackward(const std::vector<Var>& ops, const NllLossBackwardType* node) {
  // TODO(@hzfan): handle weight, reduction and ignore_index
  LTC_CHECK_EQ(ops.size(), 3U);
  Var grad_output = ops[0];
  Var logits = ops[1];
  Var labels = ops[2];
  Var normalized_dy = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.reshape"),
    {grad_output, MakeConstant(TupleInt({})), MakeConstant(Bool(false))}));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.nll_loss_dpred"),
    {normalized_dy, labels, logits}));
}

template <class NllLossBackwardType>
Var MNMNodeLowering::LowerNllLossBackward(
    const NllLossBackwardType* node) {
  // TODO(@hzfan): handle weight, reduction and ignore_index
  Var grad_output = loctx()->GetOutputOp(node->operand(0));
  Var logits = loctx()->GetOutputOp(node->operand(1));
  Var labels = loctx()->GetOutputOp(node->operand(2));
  return BuildNllLossBackward({grad_output, logits, labels}, node);
}

Var BuildExpand(const std::vector<Var>& ops, const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = ops[0];
  std::vector<int64_t> size = node->size();
  lazy_tensors::Shape shape = node->operand(0).node->shape();
  int offset = size.size() - shape.dimensions_size();
  LTC_CHECK_GE(size.size(), shape.dimensions_size());
  if (offset > 0) {
    x = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.expand_dims"),
      {x, MakeConstant(Int(0)), MakeConstant(Int(offset))}));
  }
  for (int i = 0; i < size.size(); ++i) {
    int64_t repeats = -1;
    if (i - offset < 0) {
      repeats = size[i];
    } else {
      repeats = size[i] / shape.dimensions(i - offset);
      LTC_CHECK_EQ(size[i] % shape.dimensions(i - offset), 0);
    }
    if (repeats != 1) {
      x = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.repeat"),
        {x, MakeConstant(Int(repeats)), MakeConstant(Int(i))}));
    }
  }
  return x;
}

Var MNMNodeLowering::LowerExpand(const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildExpand({x}, node);
}

Var BuildAsStridedViewUpdate(const std::vector<Var>& ops, const ir::ops::AsStridedViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  LTC_CHECK_EQ(ops.size(), node->operands().size());
  for (size_t i = 0; i < ops.size(); ++i) {
    ops[i]->checked_type_ = ToMNMType(node->operand(i).shape());
  }
  // TODO(@hzfan): allow transpose
  for (size_t i = 0; i + 1 < node->stride().size(); ++i) {
    LTC_CHECK_GE(node->stride()[i], node->stride()[i + 1]);
  }
  // TODO(@hzfan): allow offset
  LTC_CHECK_EQ(node->storage_offset(), 0);
  // TODO(@hzfan): allow update being a subarray of target
  auto target_tty = Downcast<TensorType>(ops[0]->checked_type());
  auto update_tty = Downcast<TensorType>(ops[1]->checked_type());
  size_t num_dims = target_tty->shape.size();
  LTC_CHECK_EQ(target_tty->dtype, update_tty->dtype);
  LTC_CHECK_EQ(target_tty->shape.size(), update_tty->shape.size());
  LTC_CHECK_EQ(num_dims, node->size().size());
  for (size_t i = 0; i < num_dims; ++i) {
    const auto* target_dim = target_tty->shape[i].as<IntImmNode>();
    const auto* update_dim = update_tty->shape[i].as<IntImmNode>();
    LTC_CHECK(target_dim);
    LTC_CHECK(update_dim);
    LTC_CHECK_EQ(target_dim->value, update_dim->value);
    LTC_CHECK_EQ(target_dim->value, node->size()[i]);
  }
  return ops[1];
}

Var MNMNodeLowering::LowerAsStridedViewUpdate(
    const ir::ops::AsStridedViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var target = loctx()->GetOutputOp(node->operand(0));
  Var update = loctx()->GetOutputOp(node->operand(1));
  return BuildAsStridedViewUpdate({target, update}, node);
}

Var BuildAsStrided(const std::vector<Var>& ops, const ir::ops::AsStrided* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  LTC_CHECK_EQ(ops.size(), node->operands().size());
  for (size_t i = 0; i < ops.size(); ++i) {
    ops[i]->checked_type_ = ToMNMType(node->operand(i).shape());
  }
  // TODO(@hzfan): allow transpose
  for (size_t i = 0; i + 1 < node->stride().size(); ++i) {
    LTC_CHECK_GE(node->stride()[i], node->stride()[i + 1]);
  }
  // TODO(@hzfan): allow offset
  LTC_CHECK_EQ(node->storage_offset(), 0);
  // TODO(@hzfan): allow slicing an subarray of input
  auto input_tty = Downcast<TensorType>(ops[0]->checked_type());
  size_t num_dims = input_tty->shape.size();
  LTC_CHECK_EQ(num_dims, node->size().size());
  for (size_t i = 0; i < num_dims; ++i) {
    const auto* dim = input_tty->shape[i].as<IntImmNode>();
    LTC_CHECK(dim);
    LTC_CHECK_EQ(dim->value, node->size()[i]);
  }
  return ops[0];
}

Var MNMNodeLowering::LowerAsStrided(const ir::ops::AsStrided* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var input = loctx()->GetOutputOp(node->operand(0));
  return BuildAsStrided({input}, node);
}

Var MNMNodeLowering::LowerRelayExpr(const ir::ops::RelayExpr* node) {
  // Lower(node->closure().node.get());
  Var func = loctx()->GetOutputOp(node->operand(0));
  std::vector<Expr> ops;
  for (size_t i = 1; i < node->operands().size(); ++i) {
    ops.push_back(loctx()->GetOutputOp(node->operand(i)));
  }
  return BindSymbol(mnm::ir::Call(func, ops));
}

Var MNMNodeLowering::LowerRelayFunction(const ir::ops::RelayFunction* node) {
  return BindSymbol(node->func());
}

// Var BuildTupleGetItem(const std::vector<Var>& ops, const ir::ops::TupleGetItem* node) {
//   return BindSymbol(mnm::ir::TupleGetItem(ops[0], node->index()));
// }

// Var MNMNodeLowering::LowerTupleGetItem(const ir::ops::TupleGetItem* node) {
//   Var tuple = loctx()->GetOutputOp(node->operand(0));
//   return BuildTupleGetItem({tuple}, node);
// }

// Var BuildTuple(const std::vector<Var>& ops, const ir::ops::Tuple* node) {
//   std::vector<Expr> fields;
//   fields.reserve(ops.size());
//   for (const auto& op : ops) {
//     fields.emplace_back(op);
//   }
//   return BindSymbol(mnm::ir::Tuple({BindSymbol(mnm::ir::Tuple(fields))}));
// }

// Var MNMNodeLowering::LowerTuple(const ir::ops::Tuple* node) {
//   std::vector<Var> ops;
//   for (const auto& op : node->operands()) {
//     ops.push_back(loctx()->GetOutputOp(op));
//   }
//   return BuildTuple(ops, node);
// }

Var MNMNodeLowering::LowerConvolutionOverrideable(
    const ir::ops::ConvolutionOverrideable* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Var w = loctx()->GetOutputOp(node->operand(1));
  Expr stride = MakeConstant(TupleInt(node->stride()));
  Expr padding = MakeConstant(TupleInt(node->padding()));
  Expr dilation = MakeConstant(TupleInt(node->dilation()));
  Expr groups = MakeConstant(Int(node->groups()));
  Expr layout = MakeConstant(String("NCHW"));
  Expr kernel_layout = MakeConstant(String("OIHW"));
  Expr out_layout = MakeConstant(String("NCHW"));
  bool transposed = node->transposed();
  std::vector<int64_t> output_padding = node->output_padding();
  LTC_CHECK_EQ(transposed, false);
  for (const auto& i : output_padding) {
    LTC_CHECK_EQ(i, 0);
  }
  x = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.conv2d"),
    {x, w, stride, padding, dilation, groups, layout, kernel_layout, out_layout}));
  if (node->operands().size() == 3) {
    Var bias = loctx()->GetOutputOp(node->operand(2));
    Expr axis = MakeConstant(Int(1));
    x =  BindSymbol(mnm::ir::Call(Op::Get("mnm.op.bias_add"),
      {x, bias, axis}));
  }
  return x;
}

Var BuildLogSoftmaxBackwardUseIn(const std::vector<Var>& ops, const ir::ops::LogSoftmaxBackwardUseIn* node) {
  LTC_CHECK_EQ(ops.size(), 3U);
  Var dy = ops[0], y = ops[1], x = ops[2];
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.log_softmax_dx"),
    {x, y, dy, MakeConstant(Int(node->dim()))}));
}

Var MNMNodeLowering::LowerLogSoftmaxBackwardUseIn(const ir::ops::LogSoftmaxBackwardUseIn* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) {
    ops.push_back(loctx()->GetOutputOp(op));
  }
  return BuildLogSoftmaxBackwardUseIn(ops, node);
}


// Expr Shape(const Expr& expr) {
//   static auto op_shape = Op::Get("mnm.op.shape");
//   return Call(op_shape, {expr});
// }

// Var MNMNodeLowering::LowerConvolutionBackwardOverrideable(
//     const ir::ops::ConvolutionBackwardOverrideable* node) {
//   // TODO(@hzfan): remove dependency on y
//   Var dy = loctx()->GetOutputOp(node->operand(0));
//   Var x = loctx()->GetOutputOp(node->operand(1));
//   Var w = loctx()->GetOutputOp(node->operand(2));
//   Expr stride = MakeConstant(TupleInt(node->stride()));
//   Expr padding = MakeConstant(TupleInt(node->padding()));
//   Expr dilation = MakeConstant(TupleInt(node->dilation()));
//   Expr groups = MakeConstant(Int(node->groups()));
//   Expr axis = MakeConstant(Int(1));
//   Expr keep_dims = MakeConstant(Int(0));
//   Expr exclude = MakeConstant(Bool(true));
//   bool transposed = node->transposed();
//   std::vector<int64_t> output_padding = node->output_padding();
//   LTC_CHECK_EQ(transposed, false);
//   for (const auto& i : output_padding) {
//     LTC_CHECK_EQ(i, 0);
//   }
//   Expr shape_x = BindSymbol(Shape(x));
//   Expr shape_w = BindSymbol(Shape(w));
//   Expr dx = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.conv2d_dx"),
//     {w, y, dy, shape_x, stride, padding, dilation, groups}));
//   Expr dw = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.conv2d_dx"),
//     {x, y, dy, shape_w, stride, padding, dilation, groups}));
//   Expr dbias = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.sum"),
//     {dy, axis, keep_dims, exclude}));
//   return BindSymbol(mnm::ir::Tuple(Array<Expr>({dx, dw, dbias})));
// }

Var MNMNodeLowering::LowerPermute(const ir::ops::Permute* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  Expr axes = MakeConstant(TupleInt(node->dims())); 
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.transpose"), {x, axes}));
}

Var MNMNodeLowering::LowerMm(const ir::Node* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Var y = loctx()->GetOutputOp(node->operand(1));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.matmul"), {x, y}));
}

Var MNMNodeLowering::LowerAddMatMul(const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 3) << "Unexpected number of operands";
  Var x = loctx()->GetOutputOp(node->operand(0));
  Var y = loctx()->GetOutputOp(node->operand(1));
  Var bias = loctx()->GetOutputOp(node->operand(2));
  Var mm = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.matmul"), {x, y}));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.add"), {mm, bias}));
}

Var MNMNodeLowering::LowerAdaptiveAvgPool2d(
    const ir::ops::AdaptiveAvgPool2d* node) {
  Var x = loctx()->GetOutputOp(node->operand(0));
  Expr shape = MakeConstant(TupleInt(node->output_size()));
  Expr layout = MakeConstant(String("NCHW"));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.adaptive_avg_pool2d"), {x, shape, layout}));
}

Var MNMNodeLowering::LowerGenericSlice(
    const ir::ops::GenericSlice* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  std::vector<lazy_tensors::int64> limit_indices(node->base_indices().begin(), node->base_indices().end());
  std::transform(limit_indices.begin(), limit_indices.end(), node->sizes().begin(),
                 limit_indices.begin(), std::plus<lazy_tensors::int64>());
  Expr begin = MakeConstant(TupleInt(node->base_indices()));
  Expr end = MakeConstant(TupleInt(limit_indices));
  Expr strides = MakeConstant(TupleInt(std::vector<lazy_tensors::int64>(limit_indices.size(), 1)));
  Expr slice_mode = MakeConstant(String("end"));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.strided_slice"), {x, begin, end, strides, slice_mode}));
}

Var MNMNodeLowering::LowerView(const ir::ops::View* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  Expr shape = MakeConstant(TupleInt(node->output_size()));
  Expr reverse = MakeConstant(Bool(false));
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.reshape"), {x, shape, reverse}));
}

Var BuildCast(const std::vector<Var>& ops, const ir::ops::Cast* node) {
  // TODO(@hzfan): handle node->stype() and node->dtype()
  using tvm::runtime::DLDataType2String;
  return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast"), {ops[0],
    MakeConstant(String(DLDataType2String(ToMNMDType(node->type()))))}));
}

Var MNMNodeLowering::LowerCast(const ir::ops::Cast* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  Var x = loctx()->GetOutputOp(node->operand(0));
  return BuildCast({x}, node);
}

#define DEFINE_COMPARISON_OP(name, op) \
  Var Build##name(const std::vector<Var>& ops, const ir::Node* node) { \
    LTC_CHECK_EQ(node->num_outputs(), 1); \
    ops[0]->checked_type_ = ToMNMType(node->operand(0).shape()); \
    ops[1]->checked_type_ = ToMNMType(node->operand(1).shape()); \
    Var op0, op1; \
    std::tie(op0, op1) = PromoteDType(ops[0], ops[1]); \
    return BindSymbol(mnm::ir::Call(Op::Get("mnm.op."#op), {op0, op1})); \
  } \
  Var MNMNodeLowering::Lower##name(const ir::Node* node) { \
    LTC_CHECK_EQ(node->num_outputs(), 1); \
    Var op0 = loctx()->GetOutputOp(node->operand(0)); \
    Var op1 = loctx()->GetOutputOp(node->operand(1)); \
    return Build##name({op0, op1}, node); \
  }

#define DEFINE_UNARY_OP(name, op)                              \
  Var Build##name(const std::vector<Var>& ops, const ir::Node* node) { \
    LTC_CHECK_EQ(node->num_outputs(), 1); \
    return BindSymbol(mnm::ir::Call(Op::Get("mnm.op."#op), {ops[0]})); \
  } \
  Var MNMNodeLowering::Lower##name(const ir::Node* node) { \
    Var x = loctx()->GetOutputOp(node->operand(0)); \
    return Build##name({x}, node); \
  }


DEFINE_UNARY_OP(Ceil, ceil)
DEFINE_UNARY_OP(Abs, abs);
DEFINE_UNARY_OP(Neg, negative);
DEFINE_COMPARISON_OP(Ne, not_equal)
DEFINE_COMPARISON_OP(Eq, equal)
DEFINE_COMPARISON_OP(Gt, greater)

#undef DEFINE_COMPARISON_OP
#undef DEFINE_UNARY_OP

// Var MNMNodeLowering::LowerAdaptiveAvgPool2dBackward(
//     const ir::Node* node) {
//   // TODO(@hzfan): remove dependency on shape and y
//   Var dy = loctx()->GetOutputOp(node->operand(0));
//   Var x = loctx()->GetOutputOp(node->operand(1));
//   return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.adaptive_avg_pool2d_dx"), 
//     {x, y, dy, shape})); 
// }

// Var MNMNodeLowering::LowerLogSoftmaxBackward(
//     const ir::ops::LogSoftmaxBackward* node) {
//   // TODO(@hzfan): remove dependency on x
//   Var dy = loctx()->GetOutputOp(node->operand(0));
//   Var y = loctx()->GetOutputOp(node->operand(1));
//   Expr axis = MakeConstant(Int(node->dim()));
//   return BindSymbol(mnm::ir::Call(Op::Get("mnm.op.log_softmax_dx"), 
//     {x, y, dy, axis}));
// }

Var MNMNodeLowering::LowerThresholdBackward(
    const ir::ops::ThresholdBackward* node) {
  LTC_LOG(FATAL) << "NotImplementedError";
  // TODO(@hzfan): support threshold_backward. Workaround: 
  // Options: 1) use torchscript so customized AutoDiff 2) implement it in meta 3) use compare operators here
  // LTC_CHECK_EQ(node->num_outputs(), 1);
  // Var dy = loctx()->GetOutputOp(node->operand(0));
  // Var x = loctx()->GetOutputOp(node->operand(1));
  // return 
}

// Var MNMNodeLowering::LowerConstant(const ir::ops::Constant* node) {
//   LTC_CHECK_EQ(node->num_outputs(), 1);
//   // TODO(@hzfan): get the current device from env variables
//   lazy_tensors::ComputationClient::DataPtr data = CreateTensorsData({node->value().value()}, {})[0];
//   auto* mnm_data = static_cast<torch_mnm::NeuronComputationClient::NeuronData*>(data.get());
//   return BindSymbol(MakeConstant(mnm_data->handle));
// }

Var MNMNodeLowering::LowerConstant(const ir::ops::Constant* node) {
  // TODO(@hzfan): get the current device from env variables
  // TODO(@hzfan): unify LowerConstant for meta/Sunda, meta/CPU, meta/GPU
  // TODO(@hzfan): embed NeuronTensor into constants directly
  LTC_CHECK_EQ(node->num_outputs(), 1);
  mnm::Device dev(mnm::DevType::kCPU(), 0);
  int64_t nbytes = mnm::common::shape_utils::BytesCompactTensor(
    Downcast<TensorType>(ToMNMType(node->value().shape())).as<TensorTypeNode>());
  auto buffer = memory_pool::Memory::Alloc(dev, nbytes);
  DType dtype;
  std::vector<int64_t> shape;
  std::tie(shape, dtype) = ToMNMShape(node->value().shape());
  // ts.populate_fn(ts, buffer_cpu->data, nbytes);
  // lazy_tensors::ComputationClient::DataPtr data = CreateTensorsData({node->value().value()}, {})[0];
  // auto* mnm_data = static_cast<torch_mnm::NeuronComputationClient::NeuronData*>(data.get());
  PopulateTensorBuffer(node->value().value(), node->value().shape(), buffer->data, nbytes, Device("CPU"));
  auto value = TensorValue::Assemble(dev, dtype, shape, {}, buffer->data, buffer);
  return BindSymbol(MakeConstant(value));
}

TensorValue MakeScalar(float scalar, mnm::Device to_dev) {
  float a[1] = {scalar};
  static int64_t b[1] = {1};
  DType dtype = DType(DTypeCode::kFloat(), 32, 1);
  DLTensor tensor;
  tensor.data = a;
  tensor.device = mnm::Device(DevType::kCPU(), 0);
  tensor.dtype = dtype;
  tensor.shape = b;
  tensor.ndim = 0;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty({}, dtype, to_dev);
  array.CopyFrom(&tensor);
  return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
}

Var MNMNodeLowering::LowerScalar(const ir::ops::Scalar* node) {
  using ir::ops::operator<<;
  using tvm::runtime::DLDataType2String;
  LTC_CHECK_EQ(node->num_outputs(), 1);
  TensorValue tv;
  switch (node->shape().element_type()) {
    case lazy_tensors::PrimitiveType::S64:
      tv = MakeScalar(static_cast<float>(node->value().toLong()), mnm::Device(DevType::kCPU(), 0));
      break;
    case lazy_tensors::PrimitiveType::F32:
      tv = MakeScalar(static_cast<float>(node->value().toDouble()), mnm::Device(DevType::kCPU(), 0));
      break;
    default:
      LTC_LOG(FATAL) << "Unable to lower scalar " << node->value() << " of shape "
                     << node->shape();
  }
  Expr untyped_scalar = MakeConstant(tv);
  Var scalar = BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast"), 
    {untyped_scalar, MakeConstant(String(DLDataType2String(ToMNMDType(node->shape().element_type()))))}));
  Span<const int64> dimensions = node->shape().dimensions();
  return node->shape().rank() == 0 ? scalar : BindSymbol(
    mnm::ir::Call(Op::Get("mnm.op.broadcast_to"),
    {scalar, MakeConstant(TupleInt(std::vector<int64_t>(dimensions.begin(), dimensions.end())))}));
}

lazy_tensors::Shape MNMNodeLowering::Infer(const ir::Node* node) {
  const ir::OpKind& kind = node->op();
  switch (kind.op) {
    case at::aten::relu: {
      return InferRelu(node);
    }
    case at::aten::ne: {
      return InferNe(node);
    }
    case at::aten::eq: {
      return InferEq(node);
    }
    case at::aten::gt: {
      return InferGt(node);
    }
    case at::aten::expand: {
      return InferExpand(
          ir::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand)));
    }
    case at::aten::nll_loss: {
      return InferNllLoss(
          ir::NodeCast<ir::ops::NllLoss>(node, ir::OpKind(at::aten::nll_loss)));
    }
    case at::aten::nll_loss_backward: {
      return InferNllLossBackward(ir::NodeCast<ir::ops::NllLossBackward>(
          node, ir::OpKind(at::aten::nll_loss_backward)));
    }
    case at::aten::sum: {
      return InferSum(
          ir::NodeCast<ir::ops::Sum>(node, ir::OpKind(at::aten::sum)));
    }
    case at::aten::__and__:
    case at::aten::__or__:
    case at::aten::__xor__: {
      return InferBitwise(node);
    }
    default: {
      if (kind == *ir::ops::ltc_generic_slice) {
        return InferGenericSlice(ir::NodeCast<ir::ops::GenericSlice>(
            node, *ir::ops::ltc_generic_slice));
      }
      if (kind == *ir::ops::mnm_relay_expr) {
        return InferRelayExpr(
            ir::NodeCast<ir::ops::RelayExpr>(
                node, *ir::ops::mnm_relay_expr));
      }
      if (kind == *ir::ops::mnm_relay_function) {
        return InferRelayFunction(
            ir::NodeCast<ir::ops::RelayFunction>(
                node, *ir::ops::mnm_relay_function));
      }
      // if (kind == *ir::ops::mnm_tuple_get_item) {
      //   return InferTupleGetItem(
      //       ir::NodeCast<ir::ops::TupleGetItem>(
      //           node, *ir::ops::mnm_tuple_get_item));
      // }
      // if (kind == *ir::ops::mnm_tuple) {
      //   return InferTuple(
      //       ir::NodeCast<ir::ops::Tuple>(
      //           node, *ir::ops::mnm_tuple));
      // }
      LTC_LOG(FATAL) << "Shape inference not supported for operator: " << kind;
    }
  }
}

lazy_tensors::Shape MNMNodeLowering::InferExpand(const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToMNMType(x.shape())));
  }
  Var out = BuildExpand(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape MNMNodeLowering::InferBitwise(const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 2U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToMNMType(x.shape())));
  }
  Var out = BuildBitwise(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape MNMNodeLowering::InferNllLoss(const ir::ops::NllLoss* node) {
  LTC_CHECK_EQ(node->operands().size(), 2U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToMNMType(x.shape())));
  }
  Var out = BuildNllLoss(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape MNMNodeLowering::InferNllLossBackward(const ir::ops::NllLossBackward* node) {
  LTC_CHECK_EQ(node->operands().size(), 3U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToMNMType(x.shape())));
  }
  Var out = BuildNllLossBackward(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape MNMNodeLowering::InferSum(const ir::ops::Sum* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToMNMType(x.shape())));
  }
  Var out = BuildSum(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}

lazy_tensors::Shape MNMNodeLowering::InferRelayExpr(const ir::ops::RelayExpr* node) {
  return node->operand(0).shape();
}

lazy_tensors::Shape MNMNodeLowering::InferRelayFunction(const ir::ops::RelayFunction* node) {
  LTC_LOG(FATAL) << "Should not reach here";
}

#define DEFINE_INFER_COMPARISON_OP(name) \
lazy_tensors::Shape MNMNodeLowering::Infer##name(const ir::Node* node) { \
  std::vector<Var> ops; \
  for (const auto& x : node->operands()) { \
    Var var = MakeVar("operand", ToMNMType(x.shape())); \
    ops.push_back(var); \
  } \
  Var out = Build##name(ops, node); \
  Expr body = InferType(ExtractBinding(out, ops)); \
  return ToLTCShape(body->checked_type()); \
}

DEFINE_INFER_COMPARISON_OP(Ne)
DEFINE_INFER_COMPARISON_OP(Eq)
DEFINE_INFER_COMPARISON_OP(Gt)

#undef DEFINE_INFER_COMPARISON_OP

}  // namespace

std::unique_ptr<NodeLowering> NodeLowering::Create(ir::LoweringContext* loctx) {
  return std::make_unique<compiler::MNMNodeLowering>(loctx);
}

NodeLowering* NodeLowering::Get() {
  static MNMNodeLowering* mnm_node_lowering = new MNMNodeLowering(nullptr);
  return mnm_node_lowering;
}

namespace mnm_backend {

Var LowerNodeToMNM(const ir::Node* node, MNMLoweringContext* loctx) {
  auto node_lowering = NodeLowering::Create(loctx);
  MNMNodeLowering* mnm_node_lowering =
      static_cast<MNMNodeLowering*>(node_lowering.get());
  return mnm_node_lowering->LowerToMNM(node);
}

}  // namespace mnm_backend

NodeLowering* GetMNMNodeLowering() { return NodeLowering::Get(); }

std::unique_ptr<NodeLowering> CreateMNMNodeLowering(
    ir::LoweringContext* loctx) {
  return NodeLowering::Create(loctx);
}

}  // namespace compiler
}  // namespace torch_lazy_tensors