use polars_error::PolarsResult;
use polars_plan::plans::{AExpr, Context, IR};
use polars_plan::prelude::SinkType;
use polars_utils::arena::{Arena, Node};
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey};

fn is_streamable(node: Node, arena: &Arena<AExpr>) -> bool {
    polars_plan::plans::is_streamable(node, arena, Context::Default)
}

#[recursive::recursive]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PolarsResult<PhysNodeKey> {
    let ir_node = ir_arena.get(node);
    match ir_node {
        IR::SimpleProjection { input, columns } => {
            let schema = columns.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::SimpleProjection { input, schema }))
        }

        // TODO: split partially streamable selections to avoid fallback as much as possible.
        IR::Select {
            input,
            expr,
            schema,
            ..
        } if expr.iter().all(|e| is_streamable(e.node(), expr_arena)) => {
            let selectors = expr.clone();
            let schema = schema.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::Select {
                input,
                selectors,
                schema,
                extend_original: false,
            }))
        },

        // TODO: split partially streamable selections to avoid fallback as much as possible.
        IR::HStack {
            input,
            exprs,
            schema,
            ..
        } if exprs.iter().all(|e| is_streamable(e.node(), expr_arena)) => {
            let selectors = exprs.clone();
            let schema = schema.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::Select {
                input,
                selectors,
                schema,
                extend_original: true,
            }))
        },

        IR::Filter { input, predicate } if is_streamable(predicate.node(), expr_arena) => {
            let predicate = predicate.clone();
            let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
            Ok(phys_sm.insert(PhysNode::Filter { input, predicate }))
        },

        IR::DataFrameScan {
            df,
            output_schema,
            filter,
            ..
        } => {
            if let Some(filter) = filter {
                if !is_streamable(filter.node(), expr_arena) {
                    return Ok(phys_sm.insert(PhysNode::Fallback(node)));
                }
            }

            let mut phys_node = phys_sm.insert(PhysNode::InMemorySource { df: df.clone() });

            if let Some(schema) = output_schema {
                phys_node = phys_sm.insert(PhysNode::SimpleProjection {
                    input: phys_node,
                    schema: schema.clone(),
                })
            }

            if let Some(predicate) = filter.clone() {
                phys_node = phys_sm.insert(PhysNode::Filter {
                    input: phys_node,
                    predicate,
                })
            }

            Ok(phys_node)
        },

        IR::Sink { input, payload } => {
            if *payload == SinkType::Memory {
                let schema = ir_node.schema(ir_arena).into_owned();
                let input = lower_ir(*input, ir_arena, expr_arena, phys_sm)?;
                return Ok(phys_sm.insert(PhysNode::InMemorySink { input, schema }));
            }

            todo!()
        },

        _ => Ok(phys_sm.insert(PhysNode::Fallback(node))),
    }
}
