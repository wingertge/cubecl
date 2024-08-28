use cubecl_common::operator::Operator;
use proc_macro2::Span;
use quote::{format_ident, quote, quote_spanned};
use syn::{parse_quote, spanned::Spanned, Expr, Lit, LitInt, RangeLimits, Type};

use crate::{
    expression::Expression,
    scope::{Context, ManagedVar},
};

use super::{
    branch::{expand_for_loop, expand_if, expand_loop, expand_while_loop, parse_block},
    operator::{parse_binop, parse_unop},
};

impl Expression {
    pub fn from_expr(expr: Expr, context: &mut Context) -> syn::Result<Self> {
        let result = match expr.clone() {
            Expr::Assign(assign) => {
                let span = assign.span();
                let right = Self::from_expr(*assign.right, context)?;
                Expression::Assigment {
                    span,
                    ty: right.ty(),
                    left: Box::new(Self::from_expr(*assign.left, context)?),
                    right: Box::new(right),
                }
            }
            Expr::Binary(binary) => {
                let span = binary.span();
                let left = Self::from_expr(*binary.left, context)?;
                let right = Self::from_expr(*binary.right, context)?;
                if left.is_const() && right.is_const() {
                    Expression::Verbatim {
                        tokens: quote![#expr],
                    }
                } else {
                    let ty = left.ty().or(right.ty());
                    Expression::Binary {
                        span,
                        left: Box::new(left),
                        operator: parse_binop(&binary.op)?,
                        right: Box::new(right),
                        ty,
                    }
                }
            }
            Expr::Lit(literal) => {
                let ty = lit_ty(&literal.lit)?;
                Expression::Literal {
                    span: literal.span(),
                    value: literal.lit,
                    ty,
                }
            }
            Expr::Path(path) => {
                let variable = path
                    .path
                    .get_ident()
                    .and_then(|ident| context.variable(ident));
                if let Some(ManagedVar { name, ty, is_const }) = variable {
                    if is_const {
                        Expression::ConstVariable { name, ty }
                    } else {
                        Expression::Variable {
                            span: path.span(),
                            name,
                            ty,
                        }
                    }
                } else {
                    // If it's not in the scope, it's not a managed local variable. Treat it as an
                    // external value like a Rust `const`.
                    Expression::Path { path: path.path }
                }
            }
            Expr::Unary(unary) => {
                let span = unary.span();
                let input = Self::from_expr(*unary.expr, context)?;
                let ty = input.ty();
                Expression::Unary {
                    span,
                    input: Box::new(input),
                    operator: parse_unop(&unary.op)?,
                    ty,
                }
            }
            Expr::Block(block) => {
                context.push_scope();
                let block = parse_block(block.block, context)?;
                context.pop_scope();
                block
            }
            Expr::Break(br) => Expression::Break { span: br.span() },
            Expr::Call(call) => {
                let span = call.span();
                let func = Box::new(Expression::from_expr(*call.func, context)?);
                let args = call
                    .args
                    .into_iter()
                    .map(|arg| Expression::from_expr(arg, context))
                    .collect::<Result<Vec<_>, _>>()?;
                Expression::FunctionCall { func, args, span }
            }
            Expr::MethodCall(method) => {
                let span = method.span();
                let receiver = Expression::from_expr(*method.receiver.clone(), context)?;
                let args = method
                    .args
                    .iter()
                    .map(|arg| Expression::from_expr(arg.clone(), context))
                    .collect::<Result<Vec<_>, _>>()?;
                if receiver.is_const() && args.iter().all(|arg| arg.is_const()) {
                    Expression::Verbatim {
                        tokens: quote![#method],
                    }
                } else {
                    Expression::MethodCall {
                        receiver: Box::new(receiver),
                        method: method.method,
                        args,
                        span,
                    }
                }
            }
            Expr::Cast(cast) => {
                let span = cast.span();
                let from = Expression::from_expr(*cast.expr, context)?;
                Expression::Cast {
                    from: Box::new(from),
                    to: *cast.ty,
                    span,
                }
            }
            Expr::Const(block) => Expression::Verbatim {
                tokens: quote![#block],
            },
            Expr::Continue(cont) => Expression::Continue { span: cont.span() },
            Expr::ForLoop(for_loop) => expand_for_loop(for_loop, context)?,
            Expr::While(while_loop) => expand_while_loop(while_loop, context)?,
            Expr::Loop(loop_expr) => expand_loop(loop_expr, context)?,
            Expr::If(if_expr) => expand_if(if_expr, context)?,
            Expr::Range(range) => {
                let span = range.span();
                let start = range
                    .start
                    .map(|start| Expression::from_expr(*start, context))
                    .transpose()?
                    .unwrap_or_else(|| {
                        let lit = Lit::Int(LitInt::new("0", span));
                        Expression::Literal {
                            value: lit,
                            ty: parse_quote![i32],
                            span,
                        }
                    });
                let end = range
                    .end
                    .map(|end| Expression::from_expr(*end, context))
                    .transpose()?
                    .map(Box::new);
                Expression::Range {
                    start: Box::new(start),
                    end,
                    inclusive: matches!(range.limits, RangeLimits::Closed(..)),
                    span,
                }
            }
            Expr::Field(field) => {
                let span = field.span();
                let base = Expression::from_expr(*field.base.clone(), context)?;
                Expression::FieldAccess {
                    base: Box::new(base),
                    field: field.member,
                    span,
                }
            }
            Expr::Group(group) => Expression::from_expr(*group.expr, context)?,
            Expr::Paren(paren) => Expression::from_expr(*paren.expr, context)?,
            Expr::Return(ret) => Expression::Return {
                span: ret.span(),
                expr: ret
                    .expr
                    .map(|expr| Expression::from_expr(*expr, context))
                    .transpose()?
                    .map(Box::new),
                ty: context.return_type.clone(),
            },
            Expr::Array(array) => {
                let span = array.span();
                let elements = array
                    .elems
                    .into_iter()
                    .map(|elem| Expression::from_expr(elem, context))
                    .collect::<Result<_, _>>()?;
                Expression::Array { elements, span }
            }
            Expr::Tuple(tuple) => {
                let span = tuple.span();
                let elements = tuple
                    .elems
                    .into_iter()
                    .map(|elem| Expression::from_expr(elem, context))
                    .collect::<Result<_, _>>()?;
                Expression::Tuple { elements, span }
            }
            Expr::Index(index) => {
                let span = index.span();
                let expr = Expression::from_expr(*index.expr, context)?;
                let index = Expression::from_expr(*index.index, context)?;
                if is_slice(&index) {
                    let ranges = match index {
                        Expression::Array { elements, .. } => elements.clone(),
                        Expression::Tuple { elements, .. } => elements.clone(),
                        index => vec![index],
                    };
                    Expression::Slice {
                        expr: Box::new(expr),
                        ranges,
                        span,
                    }
                } else {
                    let index = match index {
                        Expression::Array { elements, span } => {
                            generate_strided_index(&expr, elements, span)?
                        }
                        index => index,
                    };
                    Expression::Index {
                        expr: Box::new(expr),
                        index: Box::new(index),
                        span,
                    }
                }
            }
            Expr::Repeat(repeat) => {
                let span = repeat.span();
                let len = Expression::from_expr(*repeat.len, context)?;
                if !len.is_const() {
                    Err(syn::Error::new(
                        span,
                        "Array initializer length must be known at compile time",
                    ))?
                }
                Expression::ArrayInit {
                    init: Box::new(Expression::from_expr(*repeat.expr, context)?),
                    len: Box::new(len),
                    span,
                }
            }
            Expr::Let(expr) => {
                let span = expr.span();
                let elem = Expression::from_expr(*expr.expr.clone(), context)?;
                if elem.is_const() {
                    Expression::Verbatim {
                        tokens: quote![#expr],
                    }
                } else {
                    Err(syn::Error::new(
                        span,
                        "let bindings aren't yet supported at runtime",
                    ))?
                }
            }
            Expr::Match(mat) => {
                let span = mat.span();
                let elem = Expression::from_expr(*mat.expr.clone(), context)?;
                if elem.is_const() {
                    Expression::Verbatim {
                        tokens: quote![#mat],
                    }
                } else {
                    Err(syn::Error::new(
                        span,
                        "match expressions aren't yet supported at runtime",
                    ))?
                }
            }
            Expr::Macro(mac) => Expression::Verbatim {
                tokens: quote![#mac],
            },
            Expr::Struct(strct) => {
                if !strct.fields.iter().all(|field| {
                    Expression::from_expr(field.expr.clone(), context)
                        .map(|field| field.is_const())
                        .unwrap_or(false)
                }) {
                    Err(syn::Error::new_spanned(
                        strct,
                        "Struct initializers aren't supported at runtime",
                    ))?
                } else {
                    Expression::Verbatim {
                        tokens: quote![#strct],
                    }
                }
            }
            Expr::Unsafe(unsafe_expr) => {
                context.with_scope(|context| parse_block(unsafe_expr.block, context))?
            }
            Expr::Infer(_) => Expression::Verbatim { tokens: quote![_] },
            Expr::Verbatim(verbatim) => Expression::Verbatim { tokens: verbatim },
            Expr::Reference(reference) => Expression::from_expr(*reference.expr, context)?,
            Expr::Try(expr) => {
                let span = expr.span();
                let expr = Expression::from_expr(*expr.expr, context)?
                    .as_const()
                    .ok_or_else(|| syn::Error::new(span, "? Operator not supported at runtime"))?;
                Expression::Verbatim {
                    tokens: quote_spanned![span=>
                        #expr?
                    ],
                }
            }
            Expr::TryBlock(_) => Err(syn::Error::new_spanned(
                expr,
                "try_blocks is unstable and not supported in kernels",
            ))?,
            e => Err(syn::Error::new_spanned(
                expr,
                format!("Unsupported expression {e:?}"),
            ))?,
        };
        Ok(result)
    }
}

fn lit_ty(lit: &Lit) -> syn::Result<Type> {
    let res = match lit {
        Lit::Int(int) => (!int.suffix().is_empty())
            .then(|| int.suffix())
            .map(|suffix| format_ident!("{suffix}"))
            .and_then(|ident| syn::parse2(quote![#ident]).ok())
            .unwrap_or_else(|| parse_quote![i32]),
        Lit::Float(float) => (!float.suffix().is_empty())
            .then(|| float.suffix())
            .map(|suffix| format_ident!("{suffix}"))
            .and_then(|ident| syn::parse2(quote![#ident]).ok())
            .unwrap_or_else(|| parse_quote![f32]),
        Lit::Bool(_) => parse_quote![bool],
        lit => Err(syn::Error::new_spanned(
            lit,
            format!("Unsupported literal type: {lit:?}"),
        ))?,
    };
    Ok(res)
}

fn generate_strided_index(
    tensor: &Expression,
    elements: Vec<Expression>,
    span: Span,
) -> syn::Result<Expression> {
    let index_ty = elements
        .first()
        .unwrap()
        .ty()
        .unwrap_or_else(|| parse_quote![u32]);
    let strided_indices = elements.into_iter().enumerate().map(|(i, elem)| {
        let i = Lit::Int(LitInt::new(&i.to_string(), span));
        let stride = Expression::MethodCall {
            receiver: Box::new(tensor.clone()),
            method: format_ident!("stride"),
            args: vec![Expression::Literal {
                value: i,
                ty: index_ty.clone(),
                span,
            }],
            span,
        };
        Expression::Binary {
            left: Box::new(elem),
            operator: Operator::Mul,
            right: Box::new(stride),
            ty: None,
            span,
        }
    });
    let sum = strided_indices
        .reduce(|a, b| Expression::Binary {
            left: Box::new(a),
            operator: Operator::Add,
            right: Box::new(b),
            ty: None,
            span,
        })
        .unwrap();
    Ok(sum)
}

fn is_slice(index: &Expression) -> bool {
    match index {
        Expression::Range { .. } => true,
        Expression::Array { elements, .. } => elements.iter().any(is_slice),
        Expression::Tuple { elements, .. } => elements.iter().any(is_slice),
        _ => false,
    }
}
