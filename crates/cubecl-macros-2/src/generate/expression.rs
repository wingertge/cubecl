use std::num::NonZero;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Generics, Ident, Path, PathArguments, PathSegment, Type};

use crate::{expression::Expression, ir_type, prefix_ir};

impl ToTokens for Expression {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let out = match self {
            Expression::Binary {
                left,
                operator,
                right,
                span,
                ..
            } => {
                let expr_ty = prefix_ir(format_ident!("{}Expr", operator.to_string()));
                quote_spanned! {*span=>
                    #expr_ty::new(
                        #left,
                        #right
                    )
                }
            }
            Expression::Unary {
                input,
                operator,
                span,
                ..
            } => {
                let ty = prefix_ir(format_ident!("{}Expr", operator.to_string()));
                quote_spanned! {*span=>
                    #ty::new(
                        #input,
                    )
                }
            }
            Expression::Variable { name, span, .. } => {
                quote_spanned! {*span=>
                    #name.clone()
                }
            }
            Expression::FieldAccess {
                base, field, span, ..
            } => {
                let access = ir_type("FieldAccess");
                let field = match field {
                    syn::Member::Named(ident) => format_ident!("__{ident}"),
                    syn::Member::Unnamed(index) => format_ident!("__{}", index.index),
                };
                quote_spanned! {*span=>
                    #base.expand().#field()
                }
            }
            Expression::Literal { value, span, ty } => {
                quote_spanned! {*span=>
                    #value
                }
            }
            Expression::Assigment {
                left, right, span, ..
            } => {
                let ty = prefix_ir(format_ident!("Assignment"));
                quote_spanned! {*span=>
                    #ty {
                        left: #left,
                        right: #right
                    }
                }
            }
            Expression::Init {
                left,
                right,
                ty,
                span,
            } => {
                let ir_type = ir_type("Initializer");
                let ty = right.ty().map(|ty| quote![::<#ty>]);
                quote_spanned! {*span=>
                    #ir_type #ty {
                        left: #left,
                        right: #right
                    }
                }
            }
            Expression::Verbatim { tokens } => {
                let span = tokens.span();
                quote_spanned! {span=>
                    #tokens
                }
            }
            Expression::Block {
                inner,
                ret,
                ty,
                span,
            } => {
                let block = ir_type("Block");
                let ret = ret
                    .as_ref()
                    .map(|ret| quote![#ret])
                    .unwrap_or_else(|| quote![()]);
                quote_spanned! {*span=>
                    {
                        let mut __statements = Vec::new();
                        #(#inner)*
                        #block::new(__statements, #ret)
                    }
                }
            }
            Expression::FunctionCall { func, span, args } => {
                let associated_type = fn_associated_type(func);
                // We pass in the `Variable`s and `Literal`s into the expansion so they can be rebound
                // in the function root scope
                if let Some((ty_path, name)) = associated_type {
                    let static_expand = ir_type("StaticExpand");
                    quote_spanned! {*span=>
                        <#ty_path as #static_expand>::Expanded::#name(#(#args),*)
                    }
                } else {
                    let (generics, path) = split_generics(func);
                    quote_spanned! {*span=>
                        #path::expand #generics(#(#args),*)
                    }
                }
            }
            Expression::MethodCall {
                receiver,
                method,
                args,
                span,
            } => {
                let expand = if receiver.is_const() {
                    format_ident!("partial_expand")
                } else {
                    format_ident!("expand")
                };
                quote_spanned! {*span=>
                    #receiver.#expand().#method(#(#args),*)
                }
            }
            Expression::Break { span } => {
                let brk = ir_type("Break");
                quote_spanned! {*span=>
                    #brk
                }
            }
            Expression::Cast { from, to, span } => {
                let cast = ir_type("Cast");
                quote_spanned! {*span=>
                    #cast {
                        from: #from,
                        _to: PhantomData::<#to>
                    }
                }
            }
            Expression::Continue { span } => {
                let cont = ir_type("Continue");
                quote_spanned! {*span=>
                    #cont
                }
            }
            Expression::ForLoop {
                range,
                unroll,
                var_name,
                var_ty,
                var_mut,
                block,
                span,
            } => {
                let variable = generate_var(var_name, var_ty, *span, None);
                let for_ty = ir_type("ForLoop");

                if let Some(unroll) = unroll {
                    quote_spanned! {*span=>
                        {
                            let #var_name = #variable;
                            if #unroll {
                                #for_ty::new_unroll(#range, #var_name, #block)
                            } else {
                                #for_ty::new(#range, #var_name, #block)
                            }
                        }
                    }
                } else {
                    quote_spanned! {*span=>
                        {
                            let #var_name = #variable;
                            #for_ty::new(#range, #var_name, #block)
                        }
                    }
                }
            }
            Expression::ConstVariable { name, .. } => quote![#name],
            Expression::Path { path, .. } => quote![#path],
            Expression::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                let range = ir_type("RangeExpr");
                quote_spanned! {*span=>
                    #range::new(#start, #end, #inclusive)
                }
            }
        };

        tokens.extend(out);
    }
}

pub fn generate_var(
    name: &Ident,
    ty: &Option<Type>,
    span: Span,
    vectorization: Option<TokenStream>,
) -> TokenStream {
    let var = ir_type("Variable");
    let name = name.to_token_stream().to_string();
    let ty = ty.as_ref().map(|ty| {
        quote_spanned! {ty.span()=>
            ::<#ty>
        }
    });
    let vectorization = vectorization.unwrap_or(quote![None]);
    quote_spanned! {span=>
        #var #ty ::new(#name, #vectorization)
    }
}

fn fn_associated_type(path: &Expression) -> Option<(Path, PathSegment)> {
    if !matches!(path, Expression::Path { .. }) {
        panic!("path: {path:?}");
    }
    match path {
        Expression::Path { path, .. } => {
            let is_assoc = path
                .segments
                .iter()
                .nth_back(1)
                .and_then(|it| it.ident.to_string().chars().next())
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false);
            if is_assoc {
                let mut path = path.clone();
                let name = path.segments.pop().unwrap().into_value();
                path.segments.pop_punct();
                Some((path, name))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn split_generics(path: &Expression) -> (PathArguments, TokenStream) {
    let mut path = match path {
        Expression::Path { path, .. } => path.clone(),
        _ => return (PathArguments::None, quote![#path]),
    };
    let generics = if let Some(last) = path.segments.last_mut() {
        core::mem::replace(&mut last.arguments, PathArguments::None)
    } else {
        PathArguments::None
    };
    (generics, quote![#path])
}
