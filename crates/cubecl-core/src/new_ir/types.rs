use crate::ir::{Elem, FloatKind, IntKind};

use super::Expr;

pub trait SquareType {
    fn ir_type() -> Elem;
}

pub trait KernelArg {}

impl<T: SquareType> KernelArg for T {}

pub trait KernelStruct: SquareType + Sized {
    type Expanded<Base: Expr<Output = Self> + Clone>;

    fn expand<Base: Expr<Output = Self> + Clone>(base: Base) -> Self::Expanded<Base>;
}

macro_rules! primitive {
    ($primitive:ident, $var_type:expr) => {
        impl SquareType for $primitive {
            fn ir_type() -> Elem {
                $var_type
            }
        }
    };
}

primitive!(i32, Elem::Int(IntKind::I32));
primitive!(i64, Elem::Int(IntKind::I64));
primitive!(u32, Elem::UInt);
primitive!(f32, Elem::Float(FloatKind::F32));
primitive!(f64, Elem::Float(FloatKind::F64));

primitive!(bool, Elem::Bool);