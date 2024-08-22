use cubecl_core::new_ir::{Block, Statement};
use cubecl_macros_2::cube2;
use pretty_assertions::assert_eq;

mod common;
use common::*;

#[test]
fn collapses_constants() {
    #[allow(unused)]
    #[cube2]
    fn collapses_constants(#[comptime] a: u32) -> u32 {
        let b = 2;
        let c = a * b;

        let d = c + a;
        d
    }

    let expanded = collapses_constants::expand(1);
    let expected = Block::<u32>::new(vec![Statement::Return(lit(3u32))]);

    assert_eq!(expanded, expected);
}