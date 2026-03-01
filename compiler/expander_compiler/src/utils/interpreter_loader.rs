use crate::frontend::{Config, RootAPI, Variable};

pub struct M31Loader {
    symbols: Vec<Vec<Variable>>,
}

impl Default for M31Loader {
    fn default() -> Self {
        Self::new()
    }
}

impl M31Loader {
    pub fn new() -> Self {
        M31Loader { symbols: vec![] }
    }

    /// Add two m31 numbers
    pub fn big_array_add<C: Config, B: RootAPI<C>>(
        api: &mut B,
        a: &[Variable],
        b: &[Variable],
        nb_bits: usize,
    ) -> Vec<Variable> {
        if a.len() != b.len() {
            panic!("BigArrayAdd: length of a and b must be equal");
        }
        let mut c = vec![api.constant(0); a.len()];
        let mut carry = api.constant(0);
        for i in 0..a.len() {
            c[i] = api.add(a[i], b[i]);
            c[i] = api.add(c[i], carry);
            carry = api.to_binary(c[i], nb_bits + 1)[nb_bits];
            let tmp = api.mul(carry, 1 << nb_bits);
            c[i] = api.sub(c[i], tmp);
        }
        c
    }

    pub fn register_lval(&mut self, lval: usize, val: Vec<Variable>) {
        assert_eq!(lval, self.symbols.len());
        self.symbols.push(val);
    }

    pub fn get_rval_scalar(&self, rval: usize) -> Variable {
        assert_eq!(self.symbols[rval].len(), 1);
        self.symbols[rval][0]
    }

    fn parse_lval(toks: &[&str]) -> usize {
        let raw = toks[1];
        assert!((raw.chars().nth(0) == Some('=')));
        assert!((raw.chars().nth(0) == Some('v')));
        raw[2..].parse::<usize>().unwrap()
    }

    fn parse_rval_scalar<C: Config, B: RootAPI<C>>(
        &self,
        toks: &[&str],
        idx: usize,
        api: &mut B,
    ) -> Variable {
        let raw = toks[idx];
        if raw.chars().nth(0) == Some('v') {
            let value = raw[1..].parse::<i32>().unwrap();
            if value < 0 {
                panic!("negative value: {value}");
            }
            self.get_rval_scalar(value as usize)
        } else {
            let value = raw.parse::<u32>().unwrap();
            api.constant(value)
        }
    }

    fn parse_idx(toks: &[&str], idx: usize) -> usize {
        let raw = toks[idx];
        raw.parse::<usize>().unwrap()
    }

    pub fn bitwise_binary_gate<C: Config, B: RootAPI<C>>(
        &mut self,
        opcode: &str,
        lval: usize,
        lhs: Variable,
        rhs: Variable,
        api: &mut B,
    ) {
        let gate = match opcode {
            "xor" => api.xor(lhs, rhs),
            "and" => api.and(lhs, rhs),
            "or" => api.or(lhs, rhs),
            "mul" => api.mul(lhs, rhs),
            _ => {
                panic!("unknown opcode: {opcode}");
            }
        };
        self.register_lval(lval, [gate].to_vec());
    }

    pub fn load<C: Config, B: RootAPI<C>>(
        &mut self,
        fname: &str,
        input: &[Vec<Variable>],
        output: &mut Vec<Vec<Vec<Variable>>>,
        api: &mut B,
    ) {
        eprintln!("loading {fname}");
        let raw = std::fs::read_to_string(fname).unwrap();
        for line in raw.lines() {
            let v = line.split_whitespace().collect::<Vec<_>>();
            match v[0] {
                "num_args" | "input" | "output" => {}
                "decompose" => {
                    let lval = Self::parse_lval(&v);
                    let decomposed = if v[2].eq("i") {
                        assert_eq!(v[2], "i");
                        let i = Self::parse_idx(&v, 3);
                        let j = Self::parse_idx(&v, 4);
                        let nbits = Self::parse_idx(&v, 5);
                        api.new_hint("myhint.tobinary", &[input[i][j]], nbits)
                    } else {
                        let rval = self.parse_rval_scalar(&v, 2, api);
                        api.new_hint("myhint.tobinary", &[rval], 30)
                    };
                    self.register_lval(lval, decomposed);
                }
                "extractbit" => {
                    let lval = Self::parse_lval(&v);
                    let rval = v[2][1..].parse::<usize>().unwrap();
                    let i = Self::parse_idx(&v, 3);
                    self.register_lval(lval, [self.symbols[rval][i]].to_vec());
                }
                "xor" | "and" | "or" | "mul" => {
                    let lval = Self::parse_lval(&v);
                    let lhs = self.parse_rval_scalar(&v, 2, api);
                    let rhs = self.parse_rval_scalar(&v, 3, api);
                    self.bitwise_binary_gate(v[0], lval, lhs, rhs, api);
                }
                "zk.m31.compose" => {
                    let lval = Self::parse_lval(&v);
                    let n = Self::parse_idx(&v, 2);
                    let mut to_compose = vec![];
                    for i in 0..n {
                        let rval = self.parse_rval_scalar(&v, 3 + i, api);
                        to_compose.push(rval);
                    }
                    while to_compose.len() < 60 {
                        to_compose.push(api.constant(0));
                    }
                    let lo = api.from_binary(&to_compose[..30]);
                    let hi = api.from_binary(&to_compose[30..]);
                    self.register_lval(lval, [lo, hi].to_vec());
                }
                "zk.m31.add" => {
                    let lval = Self::parse_lval(&v);
                    let lhs = v[2][1..].parse::<usize>().unwrap();
                    let rhs = v[3][1..].parse::<usize>().unwrap();
                    assert_eq!(self.symbols[lhs].len(), 2);
                    assert_eq!(self.symbols[rhs].len(), 2);
                    let lhs = self.symbols[lhs].clone();
                    let rhs = self.symbols[rhs].clone();
                    let sum = Self::big_array_add(api, &lhs, &rhs, 30);
                    self.register_lval(lval, sum);
                }
                "zk.m31.extract" => {
                    let lval = Self::parse_lval(&v);
                    let rval = v[2][1..].parse::<usize>().unwrap();
                    let idx = Self::parse_idx(&v, 3);
                    self.register_lval(lval, [self.symbols[rval][idx]].to_vec());
                }
                "compose" => {
                    let lval = Self::parse_lval(&v);
                    let n = Self::parse_idx(&v, 2);
                    let mut to_compose = vec![];
                    for i in 0..n {
                        let rval = self.parse_rval_scalar(&v, 3 + i, api);
                        to_compose.push(rval);
                    }
                    let composed = api.from_binary(&to_compose);
                    self.register_lval(lval, [composed].to_vec());
                }
                "store" => {
                    assert_eq!(v[1], "o");
                    let i = Self::parse_idx(&v, 2);
                    let j = Self::parse_idx(&v, 3);
                    let idx = v[4][1..].parse::<usize>().unwrap();
                    output.resize(i + 1, vec![]);
                    output[i].resize(j + 1, vec![]);
                    output[i][j] = self.symbols[idx].clone();
                }
                _ => {
                    panic!("unknown gate type: {}", v[0]);
                }
            }
        }
    }
}
