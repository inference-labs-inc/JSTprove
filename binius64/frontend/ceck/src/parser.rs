// Copyright 2025 Irreducible Inc.
use anyhow::{Result, anyhow};
use pest::Parser;
use pest_derive::Parser;

use crate::ast::{
    AssertEqv, AssertNotEqv, Constraint, ConstraintSet, OperandExpr, ShiftOp, Term, TestFile,
    TestItem, XorExpr,
};

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct ConstraintParser;

pub fn parse_test_file(input: &str) -> Result<TestFile> {
    let mut pairs =
        ConstraintParser::parse(Rule::file, input).map_err(|e| anyhow!("Parse error: {}", e))?;

    let file = pairs.next().ok_or_else(|| anyhow!("Empty file"))?;

    let mut assertions = Vec::new();

    for pair in file.into_inner() {
        match pair.as_rule() {
            Rule::test_item => {
                let assert = parse_test_item(pair)?;
                assertions.push(assert);
            }
            Rule::EOI => {}
            _ => {
                return Err(anyhow!(
                    "Unexpected rule at top level: {:?}",
                    pair.as_rule()
                ));
            }
        }
    }

    Ok(TestFile { assertions })
}

fn parse_test_item(pair: pest::iterators::Pair<Rule>) -> Result<TestItem> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| anyhow!("Empty test item"))?;

    match inner.as_rule() {
        Rule::assert_eqv => Ok(TestItem::AssertEqv(parse_assert_eqv(inner)?)),
        Rule::assert_not_eqv => Ok(TestItem::AssertNotEqv(parse_assert_not_eqv(inner)?)),
        _ => Err(anyhow!("Unexpected test item type: {:?}", inner.as_rule())),
    }
}

fn parse_assert_eqv(pair: pest::iterators::Pair<Rule>) -> Result<AssertEqv> {
    let mut inner = pair.into_inner();

    let lhs = inner
        .next()
        .ok_or_else(|| anyhow!("Missing left-hand side constraint set in assert_eqv"))?;
    let rhs = inner
        .next()
        .ok_or_else(|| anyhow!("Missing right-hand side constraint set in assert_eqv"))?;

    Ok(AssertEqv {
        lhs: parse_constraint_set_inner(lhs)?,
        rhs: parse_constraint_set_inner(rhs)?,
    })
}

fn parse_assert_not_eqv(pair: pest::iterators::Pair<Rule>) -> Result<AssertNotEqv> {
    let mut inner = pair.into_inner();

    let lhs = inner
        .next()
        .ok_or_else(|| anyhow!("Missing left-hand side constraint set in assert_not_eqv"))?;
    let rhs = inner
        .next()
        .ok_or_else(|| anyhow!("Missing right-hand side constraint set in assert_not_eqv"))?;

    Ok(AssertNotEqv {
        lhs: parse_constraint_set_inner(lhs)?,
        rhs: parse_constraint_set_inner(rhs)?,
    })
}

fn parse_constraint_set_inner(pair: pest::iterators::Pair<Rule>) -> Result<ConstraintSet> {
    let mut constraints = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::constraint => constraints.push(parse_constraint(inner)?),
            _ => {
                return Err(anyhow!(
                    "Unexpected rule in constraint_set: {:?}",
                    inner.as_rule()
                ));
            }
        }
    }

    Ok(ConstraintSet { constraints })
}

fn parse_constraint(pair: pest::iterators::Pair<Rule>) -> Result<Constraint> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| anyhow!("Empty constraint"))?;

    match inner.as_rule() {
        Rule::and_constraint => parse_and_constraint(inner),
        Rule::mul_constraint => parse_mul_constraint(inner),
        _ => Err(anyhow!("Unexpected constraint type: {:?}", inner.as_rule())),
    }
}

fn parse_and_constraint(pair: pest::iterators::Pair<Rule>) -> Result<Constraint> {
    let mut operands = pair.into_inner();

    let a = parse_operand(
        operands
            .next()
            .ok_or_else(|| anyhow!("Missing operand A in AND constraint"))?,
    )?;
    let b = parse_operand(
        operands
            .next()
            .ok_or_else(|| anyhow!("Missing operand B in AND constraint"))?,
    )?;
    let c = parse_operand(
        operands
            .next()
            .ok_or_else(|| anyhow!("Missing operand C in AND constraint"))?,
    )?;

    Ok(Constraint::And { a, b, c })
}

fn parse_mul_constraint(pair: pest::iterators::Pair<Rule>) -> Result<Constraint> {
    let mut operands = pair.into_inner();

    let a = parse_operand(
        operands
            .next()
            .ok_or_else(|| anyhow!("Missing operand A in MUL constraint"))?,
    )?;
    let b = parse_operand(
        operands
            .next()
            .ok_or_else(|| anyhow!("Missing operand B in MUL constraint"))?,
    )?;
    let hi = parse_operand(
        operands
            .next()
            .ok_or_else(|| anyhow!("Missing operand HI in MUL constraint"))?,
    )?;
    let lo = parse_operand(
        operands
            .next()
            .ok_or_else(|| anyhow!("Missing operand LO in MUL constraint"))?,
    )?;

    Ok(Constraint::Mul { a, b, hi, lo })
}

fn parse_operand(pair: pest::iterators::Pair<Rule>) -> Result<OperandExpr> {
    match pair.as_rule() {
        Rule::xor_expr => parse_xor_expr(pair),
        Rule::term => Ok(OperandExpr::Term(parse_term(pair)?)),
        Rule::operand => {
            // operand can contain either xor_expr or term
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| anyhow!("Empty operand"))?;
            parse_operand(inner)
        }
        _ => Err(anyhow!("Unexpected operand type: {:?}", pair.as_rule())),
    }
}

fn parse_xor_expr(pair: pest::iterators::Pair<Rule>) -> Result<OperandExpr> {
    let mut terms = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::term => terms.push(parse_term(inner)?),
            _ => {
                return Err(anyhow!(
                    "Unexpected rule in xor_expr: {:?}",
                    inner.as_rule()
                ));
            }
        }
    }

    Ok(OperandExpr::Xor(XorExpr { terms }))
}

fn parse_term(pair: pest::iterators::Pair<Rule>) -> Result<Term> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| anyhow!("Empty term"))?;

    match inner.as_rule() {
        Rule::literal => parse_literal(inner),
        Rule::wire_id => Ok(Term::Wire(inner.as_str().to_string())),
        Rule::shifted_term => parse_shifted_term(inner),
        _ => Err(anyhow!("Unexpected term type: {:?}", inner.as_rule())),
    }
}

fn parse_literal(pair: pest::iterators::Pair<Rule>) -> Result<Term> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| anyhow!("Empty literal"))?;

    match inner.as_rule() {
        Rule::hex_lit => {
            let hex_str = inner.as_str().trim_start_matches("0x").replace('_', "");
            let value = u64::from_str_radix(&hex_str, 16)
                .map_err(|e| anyhow!("Invalid hex literal: {}", e))?;
            Ok(Term::Literal(value))
        }
        Rule::dec_lit => {
            let value = inner
                .as_str()
                .parse::<u64>()
                .map_err(|e| anyhow!("Invalid decimal literal: {}", e))?;
            Ok(Term::Literal(value))
        }
        _ => Err(anyhow!("Unexpected literal type: {:?}", inner.as_rule())),
    }
}

fn parse_shifted_term(pair: pest::iterators::Pair<Rule>) -> Result<Term> {
    let mut inner = pair.into_inner();

    let shift_op = inner.next().ok_or_else(|| anyhow!("Missing shift op"))?;
    let op = match shift_op.as_str() {
        "sll" => ShiftOp::Sll,
        "slr" => ShiftOp::Slr,
        "sar" => ShiftOp::Sar,
        "ror" => ShiftOp::Ror,
        _ => return Err(anyhow!("Unknown shift op: {}", shift_op.as_str())),
    };

    let wire = inner
        .next()
        .ok_or_else(|| anyhow!("Missing wire in shift"))?;
    let wire_name = wire.as_str().to_string();

    let amount = inner
        .next()
        .ok_or_else(|| anyhow!("Missing shift amount"))?;
    let amount_val = amount
        .as_str()
        .parse::<usize>()
        .map_err(|e| anyhow!("Invalid shift amount: {}", e))?;

    if amount_val >= 64 {
        return Err(anyhow!(
            "Shift amount {} is out of range (must be < 64)",
            amount_val
        ));
    }

    Ok(Term::Shifted {
        wire: wire_name,
        op,
        amount: amount_val,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_and() {
        let input = r#"
            (assert_eqv
                (constraint_set
                    (and $v0 $v1 $v2)
                )
                (constraint_set
                    (and $v0 $v1 $v2)
                )
            )
        "#;

        let result = parse_test_file(input);
        assert!(result.is_ok());
        let test_file = result.expect("parse should succeed");
        assert_eq!(test_file.assertions.len(), 1);
        match &test_file.assertions[0] {
            TestItem::AssertEqv(assert_eqv) => {
                assert_eq!(assert_eqv.lhs.constraints.len(), 1);
            }
            _ => panic!("Expected AssertEqv"),
        }
    }

    #[test]
    fn test_parse_xor_operand() {
        let input = r#"
            (assert_eqv
                (constraint_set
                    (and (xor $v0 $v1) 0xFFFFFFFF_FFFFFFFF $v2)
                )
                (constraint_set
                    (and (xor $v0 $v1) 0xFFFFFFFF_FFFFFFFF $v2)
                )
            )
        "#;

        let result = parse_test_file(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_shifted_term() {
        let input = r#"
            (assert_eqv
                (constraint_set
                    (and $v0 $v1 (xor $v2 (sll $v3 5)))
                )
                (constraint_set
                    (and $v0 $v1 (xor $v2 (sll $v3 5)))
                )
            )
        "#;

        let result = parse_test_file(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_assert_not_eqv() {
        let input = r#"
            (assert_not_eqv
                (constraint_set
                    (and $v0 $v1 $v2)
                )
                (constraint_set
                    (and $v0 $v1 0x0)
                )
            )
        "#;

        let result = parse_test_file(input);
        assert!(result.is_ok());
        let test_file = result.expect("parse should succeed");
        assert_eq!(test_file.assertions.len(), 1);
        assert!(matches!(test_file.assertions[0], TestItem::AssertNotEqv(_)));
    }

    #[test]
    fn test_parse_mixed_assertions() {
        let input = r#"
            (assert_eqv
                (constraint_set
                    (and $v0 $v1 $v2)
                )
                (constraint_set
                    (and $v0 $v1 $v2)
                )
            )
            (assert_not_eqv
                (constraint_set
                    (and $v0 $v1 $v2)
                )
                (constraint_set
                    (mul $v0 $v1 $v2 $v3)
                )
            )
        "#;

        let result = parse_test_file(input);
        assert!(result.is_ok());
        let test_file = result.expect("parse should succeed");
        assert_eq!(test_file.assertions.len(), 2);
        assert!(matches!(test_file.assertions[0], TestItem::AssertEqv(_)));
        assert!(matches!(test_file.assertions[1], TestItem::AssertNotEqv(_)));
    }
}
