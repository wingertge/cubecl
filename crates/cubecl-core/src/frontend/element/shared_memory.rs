use std::{
    marker::PhantomData,
    num::NonZero,
    ops::{Index, IndexMut},
};

use crate::{
    frontend::CubeContext,
    ir::Elem,
    new_ir::{
        flatten::item, Container, Expr, Expression, IndexExpr, SliceExpr, SliceRangeExpr,
        SquareType, Strided, Vectorization,
    },
    prelude::*,
    unexpanded,
};

use super::{Dim1, ExpandElement, Integer, Primitive, Slice};

#[derive(Clone, Copy, Expand)]
pub struct SharedMemory<T: SquareType> {
    _val: PhantomData<T>,
}

impl<T: SquareType> Strided for SharedMemory<T> {
    type Dims = Dim1;
}

impl<T: SquareType> Container for SharedMemory<T> {
    type Item = T;
}

#[derive(Clone, Debug, PartialEq)]
pub enum SharedMemoryExpr {
    Init {
        size: u32,
        ty: Elem,
        vectorization: Vectorization,
    },
}

impl SharedMemoryExpr {
    pub fn ir_type(&self) -> Elem {
        match self {
            SharedMemoryExpr::Init { ty, .. } => *ty,
        }
    }

    pub fn vectorization(&self) -> Vectorization {
        match self {
            SharedMemoryExpr::Init { vectorization, .. } => *vectorization,
        }
    }

    pub fn flatten(self, context: &mut CubeContext) -> Option<ExpandElement> {
        match self {
            SharedMemoryExpr::Init {
                size,
                ty,
                vectorization,
            } => {
                let var = context.create_shared(item(ty, vectorization), size);
                var.into()
            }
        }
    }
}

#[derive(new)]
pub struct SharedMemoryInit<T: SquareType> {
    pub size: u32,
    pub vectorization: Vectorization,
    pub _type: PhantomData<T>,
}

impl<T: SquareType> Expr for SharedMemoryInit<T> {
    type Output = SharedMemory<T>;

    fn expression_untyped(&self) -> Expression {
        SharedMemoryExpr::Init {
            size: self.size,
            ty: T::ir_type(),
            vectorization: self.vectorization(),
        }
        .into()
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}

#[expand_impl]
impl<T: Primitive> SharedMemory<T> {
    pub fn new(_size: u32) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn vectorized(_size: u32, _vectorization_factor: u8) -> Self {
        SharedMemory { _val: PhantomData }
    }

    #[expanded]
    pub fn vectorized(size: u32, vectorization_factor: u8) -> impl Expr<Output = SharedMemory<T>> {
        SharedMemoryInit::new(size, NonZero::new(vectorization_factor))
    }

    #[expanded]
    pub fn new(size: u32) -> impl Expr<Output = SharedMemory<T>> {
        SharedMemoryInit::new(size, None)
    }

    #[expanded]
    pub fn index<Idx: Expr>(self, index: Idx) -> impl Expr<Output = T>
    where
        Idx::Output: Integer,
    {
        IndexExpr::new(self.0, index)
    }

    #[expanded]
    pub fn slice<TNum: Integer>(
        self,
        ranges: Vec<Box<dyn Expr<Output = SliceRangeExpr<TNum>>>>,
    ) -> impl Expr<Output = Slice<__Inner, TNum>> {
        SliceExpr::new(self.0, ranges)
    }
}

impl<T: SquareType, I: Integer> Index<I> for SharedMemory<T> {
    type Output = T;

    fn index(&self, _index: I) -> &Self::Output {
        unexpanded!()
    }
}

impl<T: SquareType, I: Integer> IndexMut<I> for SharedMemory<T> {
    fn index_mut(&mut self, _index: I) -> &mut Self::Output {
        unexpanded!()
    }
}
