use cubecl_core::ir::{
    Branch, ConstantScalarValue, Metadata, Operation, Operator, UnaryOperator, Variable,
};

use crate::{AtomicCounter, Optimizer, Slice};

use super::{get_out, OptimizerPass};

/// Simplifies certain expressions where one operand is constant.
/// For example: `out = x * 1` to `out = x`
pub struct ConstOperandSimplify;

impl OptimizerPass for ConstOperandSimplify {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.program.node_indices().collect::<Vec<_>>() {
            let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

            for idx in ops {
                let op = &mut opt.program[node].ops.borrow_mut()[idx];
                match op {
                    // 0 * x == 0
                    Operation::Operator(Operator::Mul(bin_op)) if bin_op.lhs.is_constant(0) => {
                        *op = assign(bin_op.out, bin_op.lhs);
                        changes.inc();
                    }
                    // x * 0 == 0
                    Operation::Operator(Operator::Mul(bin_op)) if bin_op.rhs.is_constant(0) => {
                        *op = assign(bin_op.out, bin_op.rhs);
                        changes.inc();
                    }
                    // 1 * x == x
                    Operation::Operator(Operator::Mul(bin_op)) if bin_op.lhs.is_constant(1) => {
                        *op = assign(bin_op.out, bin_op.rhs);
                        changes.inc();
                    }
                    // x * 1 == x
                    Operation::Operator(Operator::Mul(bin_op)) if bin_op.rhs.is_constant(1) => {
                        *op = assign(bin_op.out, bin_op.lhs);
                        changes.inc();
                    }
                    // 0 + x = x
                    Operation::Operator(Operator::Add(bin_op)) if bin_op.lhs.is_constant(0) => {
                        *op = assign(bin_op.out, bin_op.rhs);
                        changes.inc();
                    }
                    // x + 0 = x
                    Operation::Operator(Operator::Add(bin_op)) if bin_op.rhs.is_constant(0) => {
                        *op = assign(bin_op.out, bin_op.lhs);
                        changes.inc();
                    }
                    // x - 0 == x
                    Operation::Operator(Operator::Sub(bin_op)) if bin_op.rhs.is_constant(0) => {
                        *op = assign(bin_op.out, bin_op.lhs);
                        changes.inc();
                    }
                    // x / 1 == x, 0 / x == 0
                    Operation::Operator(Operator::Div(bin_op))
                        if bin_op.lhs.is_constant(0) || bin_op.rhs.is_constant(1) =>
                    {
                        *op = assign(bin_op.out, bin_op.lhs);
                        changes.inc();
                    }
                    // x % 1 == 0, 0 % x == 0
                    Operation::Operator(Operator::Modulo(bin_op))
                        if bin_op.rhs.is_constant(1) || bin_op.lhs.is_constant(0) =>
                    {
                        let value = ConstantScalarValue::UInt(0).cast_to(bin_op.out.item().elem());
                        *op = assign(bin_op.out, Variable::ConstantScalar(value));
                        changes.inc();
                    }
                    // true || x == true, x || true == true
                    Operation::Operator(Operator::Or(bin_op))
                        if bin_op.lhs.is_true() || bin_op.rhs.is_true() =>
                    {
                        *op = assign(bin_op.out, true.into());
                        changes.inc();
                    }
                    // false || x == x, x || false == x
                    Operation::Operator(Operator::Or(bin_op)) if bin_op.lhs.is_false() => {
                        *op = assign(bin_op.out, bin_op.rhs);
                        changes.inc();
                    }
                    // x || false == x
                    Operation::Operator(Operator::Or(bin_op)) if bin_op.rhs.is_false() => {
                        *op = assign(bin_op.out, bin_op.lhs);
                        changes.inc();
                    }
                    // false && x == false, x && false == false
                    Operation::Operator(Operator::And(bin_op))
                        if bin_op.lhs.is_false() || bin_op.rhs.is_false() =>
                    {
                        *op = assign(bin_op.out, false.into());
                        changes.inc();
                    }
                    // true && x == x
                    Operation::Operator(Operator::And(bin_op)) if bin_op.lhs.is_true() => {
                        *op = assign(bin_op.out, bin_op.rhs);
                        changes.inc();
                    }
                    // x && true == x
                    Operation::Operator(Operator::And(bin_op)) if bin_op.rhs.is_true() => {
                        *op = assign(bin_op.out, bin_op.lhs);
                        changes.inc();
                    }
                    // select(true, a, b) == a
                    Operation::Branch(Branch::Select(select)) if select.cond.is_true() => {
                        *op = assign(select.out, select.then);
                        changes.inc();
                    }
                    // select(false, a, b) == b
                    Operation::Branch(Branch::Select(select)) if select.cond.is_false() => {
                        *op = assign(select.out, select.or_else);
                        changes.inc();
                    }
                    // Constant length to const value
                    Operation::Metadata(Metadata::Length { var, out }) => match var {
                        Variable::ConstantArray { length, .. }
                        | Variable::SharedMemory { length, .. }
                        | Variable::LocalArray { length, .. } => {
                            *op = assign(*out, (*length).into());
                            changes.inc();
                        }
                        Variable::Slice { id, depth, .. } => {
                            let slice = opt.program.slices.get(&(*id, *depth));
                            if let Some(Slice {
                                const_len: Some(len),
                                ..
                            }) = slice
                            {
                                *op = assign(*out, (*len).into());
                                changes.inc();
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
}

fn assign(out: Variable, value: Variable) -> Operation {
    Operation::Operator(Operator::Assign(UnaryOperator { input: value, out }))
}

/// Evaluates expressions where both operands are constant and replaces them with simple constant
/// assignments. This can often be applied as a result of assignment merging or constant operand
/// simplification.
pub struct ConstEval;

impl OptimizerPass for ConstEval {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.node_ids() {
            let ops = opt.program[node].ops.clone();
            for op in ops.borrow_mut().values_mut() {
                let out = get_out(opt, op);
                if let Some(out) = out {
                    if let Some(const_eval) = try_const_eval(op) {
                        let input = Variable::ConstantScalar(const_eval);
                        *op = Operator::Assign(UnaryOperator { out, input }).into();
                        changes.inc();
                    }
                }
            }
        }
    }
}

macro_rules! const_eval {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs $op rhs, kind),
                (Float(lhs, kind), Float(rhs, _)) => ConstantScalarValue::Float(lhs $op rhs, kind),
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($lhs:expr, $rhs:expr; $op:path) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int($op(lhs, rhs), kind),
                (Float(lhs, kind), Float(rhs, _)) => ConstantScalarValue::Float($op(lhs, rhs), kind),
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt($op(lhs, rhs)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_int {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs $op rhs, kind),
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_float {
    ($lhs:expr, $rhs:expr; $fn:path) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Float(lhs, kind), Float(rhs, _)) => {
                    ConstantScalarValue::Float($fn(lhs, rhs), kind)
                }
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($input:expr; $fn:path) => {{
        use ConstantScalarValue::*;

        if let Some(input) = $input.as_const() {
            Some(match input {
                Float(input, kind) => ConstantScalarValue::Float($fn(input), kind),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_cmp {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Int(lhs, _), Int(rhs, _)) => ConstantScalarValue::Bool(lhs $op rhs),
                (Float(lhs, _), Float(rhs, _)) => ConstantScalarValue::Bool(lhs $op rhs),
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_bool {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Bool(lhs), Bool(rhs)) => ConstantScalarValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

fn try_const_eval(op: &mut Operation) -> Option<ConstantScalarValue> {
    let op = match op {
        Operation::Operator(operator) => operator,
        _ => return None,
    };

    match op {
        Operator::Add(op) => const_eval!(+ op.lhs, op.rhs),
        Operator::Sub(op) => const_eval!(-op.lhs, op.rhs),
        Operator::Mul(op) => const_eval!(*op.lhs, op.rhs),
        Operator::Div(op) => const_eval!(/ op.lhs, op.rhs),
        Operator::Powf(op) => const_eval_float!(op.lhs, op.rhs; num::Float::powf),
        Operator::Equal(op) => const_eval_cmp!(== op.lhs, op.rhs),
        Operator::NotEqual(op) => const_eval_cmp!(!= op.lhs, op.rhs),
        Operator::Lower(op) => const_eval_cmp!(< op.lhs, op.rhs),
        Operator::Greater(op) => const_eval_cmp!(> op.lhs, op.rhs),
        Operator::LowerEqual(op) => const_eval_cmp!(<= op.lhs, op.rhs),
        Operator::GreaterEqual(op) => const_eval_cmp!(>= op.lhs, op.rhs),
        Operator::Modulo(op) => const_eval!(% op.lhs, op.rhs),
        Operator::And(op) => const_eval_bool!(&&op.lhs, op.rhs),
        Operator::Or(op) => const_eval_bool!(|| op.lhs, op.rhs),
        Operator::Max(op) => {
            use ConstantScalarValue::*;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(lhs.elem());
                Some(match (lhs, rhs) {
                    (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs.max(rhs), kind),
                    (Float(lhs, kind), Float(rhs, _)) => {
                        ConstantScalarValue::Float(lhs.max(rhs), kind)
                    }
                    (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs.max(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Operator::Min(op) => {
            use ConstantScalarValue::*;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(lhs.elem());
                Some(match (lhs, rhs) {
                    (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs.min(rhs), kind),
                    (Float(lhs, kind), Float(rhs, _)) => {
                        ConstantScalarValue::Float(lhs.min(rhs), kind)
                    }
                    (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs.min(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Operator::BitwiseAnd(op) => const_eval_int!(&op.lhs, op.rhs),
        Operator::BitwiseOr(op) => const_eval_int!(| op.lhs, op.rhs),
        Operator::BitwiseXor(op) => const_eval_int!(^ op.lhs, op.rhs),
        Operator::ShiftLeft(op) => const_eval_int!(<< op.lhs, op.rhs),
        Operator::ShiftRight(op) => const_eval_int!(>> op.lhs, op.rhs),
        Operator::Dot(op) => const_eval!(*op.lhs, op.rhs),

        Operator::Abs(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(input.abs(), kind),
                Float(input, kind) => ConstantScalarValue::Float(input.abs(), kind),
                _ => unreachable!(),
            })
        }
        Operator::Exp(op) => const_eval_float!(op.input; num::Float::exp),
        Operator::Log(op) => const_eval_float!(op.input; num::Float::ln),
        Operator::Log1p(op) => const_eval_float!(op.input; num::Float::ln_1p),
        Operator::Cos(op) => const_eval_float!(op.input; num::Float::cos),
        Operator::Sin(op) => const_eval_float!(op.input; num::Float::sin),
        Operator::Tanh(op) => const_eval_float!(op.input; num::Float::tanh),
        Operator::Sqrt(op) => const_eval_float!(op.input; num::Float::sqrt),
        Operator::Round(op) => const_eval_float!(op.input; num::Float::round),
        Operator::Floor(op) => const_eval_float!(op.input; num::Float::floor),
        Operator::Ceil(op) => const_eval_float!(op.input; num::Float::ceil),
        Operator::Recip(op) => const_eval_float!(op.input; num::Float::recip),
        Operator::Not(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Bool(input) => ConstantScalarValue::Bool(!input),
                _ => unreachable!(),
            })
        }
        Operator::Neg(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(-input, kind),
                Float(input, kind) => ConstantScalarValue::Float(-input, kind),
                _ => unreachable!(),
            })
        }

        Operator::Fma(op) => {
            use ConstantScalarValue::*;
            let a = op.a.as_const();
            let b = op.b.as_const();
            let c = op.c.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(a.elem());
                let c = c.cast_to(a.elem());
                match (a, b, c) {
                    (Float(a, kind), Float(b, _), Float(c, _)) => {
                        ConstantScalarValue::Float(a * b + c, kind)
                    }
                    _ => unreachable!(),
                }
            })
        }
        Operator::Clamp(op) => {
            use ConstantScalarValue::*;
            let a = op.input.as_const();
            let b = op.min_value.as_const();
            let c = op.max_value.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(a.elem());
                let c = c.cast_to(a.elem());
                match (a, b, c) {
                    (Int(a, kind), Int(b, _), Int(c, _)) => {
                        ConstantScalarValue::Int(a.clamp(b, c), kind)
                    }
                    (Float(a, kind), Float(b, _), Float(c, _)) => {
                        ConstantScalarValue::Float(a.clamp(b, c), kind)
                    }
                    (UInt(a), UInt(b), UInt(c)) => ConstantScalarValue::UInt(a.clamp(b, c)),
                    _ => unreachable!(),
                }
            })
        }
        _ => None,
    }
}
