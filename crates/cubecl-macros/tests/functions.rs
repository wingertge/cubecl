use cubecl_core as cubecl;
use cubecl_core::{ir::Elem, new_ir::*, prelude::*};
use pretty_assertions::assert_eq;

mod common;
use common::*;

#[cube]
fn helper_fn(a: u32) -> u32 {
    a * 2
}

#[test]
fn function_call() {
    #[allow(unused)]
    #[cube]
    fn function_call(a: u32) -> u32 {
        helper_fn(a)
    }

    let expanded = function_call::expand(Variable::new("a", false, None)).expression_untyped();
    let expected = block_expr(
        vec![],
        Some(block_expr(
            vec![],
            Some(Expression::Binary {
                left: var_expr("a", false, Elem::UInt),
                operator: Operator::Mul,
                right: Box::new(lit(2u32)),
                vectorization: None,
                ty: Elem::UInt,
            }),
        )),
    );

    assert_eq!(expanded, expected);
}

#[derive(Expand)]
struct Dummy {
    a: u32,
}

#[expand_impl]
impl Dummy {
    fn method(&self, b: u32) -> u32 {
        self.a * b
    }

    #[expanded]
    pub fn method<B: Expr<Output = u32>>(self, b: B) -> impl Expr<Output = u32> {
        MulExpr::new(self.0.expand().__a(), b)
    }
}

#[test]
fn method_call() {
    #[allow(unused)]
    #[cube]
    fn method_call(a: Dummy) -> u32 {
        a.method(2)
    }

    let expanded = method_call::expand(Variable::new("a", false, None)).expression_untyped();
    let expected = block_expr(
        vec![],
        Some(Expression::Binary {
            left: Box::new(Expression::FieldAccess {
                base: var_expr("a", false, Elem::Unit),
                name: "a".to_string(),
                vectorization: None,
                ty: Elem::UInt,
            }),
            operator: Operator::Mul,
            right: Box::new(lit(2u32)),
            vectorization: None,
            ty: Elem::UInt,
        }),
    );

    assert_eq!(expanded, expected);
}

impl StaticExpand for Dummy {
    type Expanded = DummyExpand<Self>;
}

#[expand_impl]
impl Dummy {
    fn associated(b: u32) -> u32 {
        b * 2
    }

    #[expanded]
    pub fn associated<B: Expr<Output = u32>>(b: B) -> impl Expr<Output = u32> {
        MulExpr::new(b, 2)
    }
}

#[test]
fn associated_call() {
    #[allow(unused)]
    #[cube]
    fn associated_call() -> u32 {
        Dummy::associated(4)
    }

    let expanded = associated_call::expand().expression_untyped();
    let expected = block_expr(
        vec![],
        Some(Expression::Binary {
            left: Box::new(lit(4u32)),
            operator: Operator::Mul,
            right: Box::new(lit(2u32)),
            vectorization: None,
            ty: Elem::UInt,
        }),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn trait_functions() {
    #[cube]
    fn trait_functions<T: BitCast<u32>>() -> T {
        T::bitcast_from(1)
    }

    let expanded = trait_functions::expand::<f32>().expression_untyped();
    let expected = block_expr(
        vec![],
        Some(Expression::Binary {
            left: Box::new(lit(4u32)),
            operator: Operator::Mul,
            right: Box::new(lit(2u32)),
            vectorization: None,
            ty: Elem::UInt,
        }),
    );

    assert_eq!(expanded, expected);
}
