use std::fmt::Display;

use crate::shared::{Component, Elem, FmtLeft};

use super::{Dialect, Item, Variable};

#[derive(Clone, Debug)]
pub enum WarpInstruction<D: Dialect> {
    ReduceSum {
        input: Variable<D>,
        out: Variable<D>,
    },
    InclusiveSum {
        input: Variable<D>,
        out: Variable<D>,
    },
    ExclusiveSum {
        input: Variable<D>,
        out: Variable<D>,
    },
    ReduceProd {
        input: Variable<D>,
        out: Variable<D>,
    },
    InclusiveProd {
        input: Variable<D>,
        out: Variable<D>,
    },
    ExclusiveProd {
        input: Variable<D>,
        out: Variable<D>,
    },
    ReduceMax {
        input: Variable<D>,
        out: Variable<D>,
    },
    ReduceMin {
        input: Variable<D>,
        out: Variable<D>,
    },
    Elect {
        out: Variable<D>,
    },
    All {
        input: Variable<D>,
        out: Variable<D>,
    },
    Any {
        input: Variable<D>,
        out: Variable<D>,
    },
    Ballot {
        input: Variable<D>,
        out: Variable<D>,
    },
    Broadcast {
        input: Variable<D>,
        id: Variable<D>,
        out: Variable<D>,
    },
}

impl<D: Dialect> Display for WarpInstruction<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarpInstruction::ReduceSum { input, out } => reduce_operator(f, input, out, "+="),
            WarpInstruction::ReduceProd { input, out } => reduce_operator(f, input, out, "*="),
            WarpInstruction::ReduceMax { input, out } => reduce_comparison(f, input, out, "max"),
            WarpInstruction::ReduceMin { input, out } => reduce_comparison(f, input, out, "min"),
            WarpInstruction::All { input, out } => reduce_quantifier(f, input, out, D::warp_all),
            WarpInstruction::Any { input, out } => reduce_quantifier(f, input, out, D::warp_any),
            WarpInstruction::Ballot { input, out } => {
                assert_eq!(
                    input.item().vectorization,
                    1,
                    "Ballot can't support vectorized input"
                );
                let rhs = D::warp_ballot(&format!("{input}"));
                let out_fmt = out.fmt_left();
                write!(
                    f,
                    "
{out_fmt} = {{ {rhs}, 0, 0, 0 }};
            "
                )
            }
            WarpInstruction::Broadcast { input, id, out } => reduce_broadcast(f, input, out, id),
            WarpInstruction::Elect { out } => write!(
                f,
                "
unsigned int mask = __activemask();
unsigned int leader = __ffs(mask) - 1;
{out} = threadIdx.x % warpSize == leader;
            "
            ),
            WarpInstruction::InclusiveSum { input, out } => reduce_inclusive(f, input, out, "+="),
            WarpInstruction::InclusiveProd { input, out } => reduce_inclusive(f, input, out, "*="),
            WarpInstruction::ExclusiveSum { input, out } => {
                reduce_exclusive(f, input, out, "+=", "0")
            }
            WarpInstruction::ExclusiveProd { input, out } => {
                reduce_exclusive(f, input, out, "*=", "1")
            }
        }
    }
}

fn reduce_operator<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    op: &str,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();

    reduce_with_loop(f, input, out, acc_item, |acc, index| {
        let acc_indexed = maybe_index(acc, index);
        let shfl_xor = D::warp_shuffle_xor(&acc_indexed, "offset");
        format!("{acc_indexed} {op} {shfl_xor};")
    })
}

fn reduce_inclusive<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    op: &str,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();

    reduce_with_loop(f, input, out, acc_item, |acc, index| {
        let acc_indexed = maybe_index(acc, index);
        let shfl_up = D::warp_shuffle_up(&acc_indexed, "offset");
        let tmp = Variable::tmp(Item::scalar(acc_item.elem));
        let lane_id = Variable::<D>::ThreadIdxWarp;
        format!(
            "
{} = {shfl_up};
if({lane_id} >= offset) {{
    {acc_indexed} {op} {tmp};
}}
",
            tmp.fmt_left()
        )
    })
}

fn reduce_exclusive<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    op: &str,
    default: &str,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();

    let inclusive = Variable::tmp(acc_item);
    reduce_inclusive(f, input, &inclusive, op)?;
    let shfl = Variable::tmp(acc_item);
    writeln!(f, "{} = {{", shfl.fmt_left())?;
    for k in 0..acc_item.vectorization {
        let inclusive_indexed = maybe_index(&inclusive, k);
        writeln!(
            f,
            "{},",
            D::warp_shuffle_up(&inclusive_indexed.to_string(), "1")
        )?;
    }
    writeln!(f, "}};")?;
    let lane_id = Variable::<D>::ThreadIdxWarp;

    write!(
        f,
        "{} = ({lane_id} == 0) ? {}{{",
        out.fmt_left(),
        out.item(),
    )?;
    for _ in 0..out.item().vectorization {
        write!(f, "{default},")?;
    }
    writeln!(f, "}} : {};", cast(&shfl, out.item()))
}

fn reduce_comparison<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    cmp: &str,
) -> core::fmt::Result {
    let in_optimized = input.optimized();
    let acc_item = in_optimized.item();
    let instruction = match in_optimized.elem() {
        Elem::F16 | Elem::BF16 => format!("__h{cmp}"),
        Elem::F162 | Elem::BF162 => format!("__h{cmp}2"),
        _ => cmp.to_string(),
    };

    reduce_with_loop(f, input, out, acc_item, |acc, index| {
        let acc_indexed = maybe_index(acc, index);
        let shfl_xor = D::warp_shuffle_xor(&acc_indexed, "offset");
        format!("{acc_indexed} = {instruction}({acc_indexed}, {shfl_xor});")
    })
}

fn reduce_broadcast<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    id: &Variable<D>,
) -> core::fmt::Result {
    let rhs = (0..input.item().vectorization)
        .map(|k| D::warp_shuffle(&format!("{}", input.index(k)), &format!("{id}")))
        .collect::<Vec<_>>()
        .join(",");
    let out_fmt = out.fmt_left();
    writeln!(f, "{out_fmt} = {{ {rhs} }};")
}

fn reduce_with_loop<D: Dialect, I: Fn(&Variable<D>, usize) -> String>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    acc_item: Item<D>,
    instruction: I,
) -> core::fmt::Result {
    let acc = Variable::Named {
        name: "acc",
        item: acc_item,
    };

    writeln!(f, "auto plane_{out} = [&]() -> {} {{", out.item())?;
    writeln!(f, "    {} {} = {};", acc_item, acc, cast(input, acc_item))?;
    writeln!(
        f,
        "    for (int offset = 1; offset < warpSizeChecked; offset *=2 ) {{"
    )?;
    for k in 0..acc_item.vectorization {
        writeln!(f, "        {}", instruction(&acc, k))?;
    }
    writeln!(f, "    }};")?;
    writeln!(f, "    return {};", cast(&acc, out.item()))?;
    writeln!(f, "}};")?;
    writeln!(f, "{} = plane_{}();", out.fmt_left(), out)
}

fn reduce_quantifier<D: Dialect, Q: Fn(&str) -> String>(
    f: &mut core::fmt::Formatter<'_>,
    input: &Variable<D>,
    out: &Variable<D>,
    quantifier: Q,
) -> core::fmt::Result {
    let rhs = (0..input.item().vectorization)
        .map(|k| quantifier(&format!("{}", input.index(k))))
        .collect::<Vec<_>>()
        .join(",");
    let out_fmt = out.fmt_left();
    writeln!(f, "{out_fmt} = {{ {rhs} }};")
}

fn cast<D: Dialect>(input: &Variable<D>, target: Item<D>) -> String {
    if target != input.item() {
        let qualifier = input.const_qualifier();
        format!("reinterpret_cast<{}{}&>({})", target, qualifier, input)
    } else {
        format!("{}", input)
    }
}

fn maybe_index<D: Dialect>(var: &Variable<D>, k: usize) -> String {
    if var.item().vectorization > 1 {
        format!("{var}.i_{k}")
    } else {
        format!("{var}")
    }
}
