
/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/wrap_identity.cc
 * \brief Wrap identity output values with mnm.op.copy
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/binding.h"
#include "meta/src/pass/common.h"
#include "meta/src/pass/let_list.h"

namespace mnm {

namespace binding {

using namespace mnm::ir;
using namespace mnm::value;

TensorValue MakeOnes(Device to_dev);

}  // namespace binding

namespace pass {
namespace wrap_identity {

using namespace mnm::ir;
using namespace mnm::op;

class IdentityWrapper : ExprMutator {
 public:
  IdentityWrapper() { }

  Expr VisitExpr_(const VarNode* node) {
    const static Op copy = Op::Get("mnm.op.copy");
    const static Op add = Op::Get("mnm.op.add");
    const static Op multiply = Op::Get("mnm.op.multiply");
    const static Op reshape = Op::Get("mnm.op.reshape");
    const static Op cast_like = Op::Get("mnm.op.cast_like");
    const static Op negative = Op::Get("mnm.op.negative");
    if (params_.find(GetRef<Var>(node)) != params_.end()) {
      // Var v1 = MakeVar("neg" + std::to_string(cnt_++), {});
      // Var v2 = MakeVar("neg" + std::to_string(cnt_++), {});
      // return ll_->Push(v1, Call(negative, {ll_->Push(v2,
      //   Call(negative, {GetRef<Var>(node)}))}));
      Var v1 = MakeVar("neg" + std::to_string(cnt_++), {});
      return ll_->Push(v1, Call(copy, {GetRef<Var>(node)}));
    }
    return ExprMutator::VisitExpr_(node);
  }

  Expr operator() (const Expr& e) {
    auto func = Downcast<Function>(e);
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(func->body);
    std::vector<Var> vars = ell->vars;
    std::vector<Expr> exprs = ell->exprs;
    size_t n = vars.size();
    CHECK_EQ(vars.size(), exprs.size());
    for (const auto& param : func->params) {
      params_.insert(param);
    }
    Expr body = LetList::With([&](LetList* ll) {
      ll_ = ll;
      for (size_t i = 0; i + 1 < n; ++i) {
        ll->Push(vars[i], exprs[i]);
      }
      ll->Push(vars[n - 1], VisitExpr(exprs[n - 1]));
      return ell->ret;
    });
    return Function(func->params, body, func->ret_type, func->type_params);
  }

 private:
  LetList* ll_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> params_;
  int cnt_{0};
};

}  // namespace wrap_identity

Pass WrapIdentity() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(wrap_identity::IdentityWrapper()(f));
  };
  return CreateMNMFunctionPass(pass_func, 1, "WrapIdentity", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.WrapIdentity").set_body_typed(WrapIdentity);

}  // namespace pass
}  // namespace mnm
