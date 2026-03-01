use crate::{
    circuit::input_mapping::{InputMapping, EMPTY},
    frontend::CircuitField,
    utils::{bucket_sort::bucket_sort, static_hash_map::StaticHashMap},
};

use super::{
    Circuit, Constraint, Debug, FieldArith, Hash, HashMap, HashSet, Instruction, IrConfig,
    RootCircuit,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Element {
    circuit_id: usize,
    id: usize,
    typ: ElementType,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum ElementType {
    Var,
    Insn,
    Output,
    CircuitConstrained,
}
use ElementType::*;

impl<Irc: IrConfig> RootCircuit<Irc> {
    pub fn remove_unreachable(&self) -> (Self, InputMapping) {
        let order = self.topo_order();
        // first, remove unused sub circuits based on constraints
        let mut circuit_used: HashSet<usize> = HashSet::new();
        circuit_used.insert(0);
        for circuit_id in order.iter() {
            if circuit_used.contains(circuit_id) {
                let circuit = self.circuits.get(circuit_id).unwrap();
                for insn in circuit.instructions.iter() {
                    if let Some((sub_circuit_id, _, _)) = insn.as_sub_circuit_call() {
                        circuit_used.insert(sub_circuit_id);
                    }
                }
            }
        }
        // now the order only contains used circuits
        let order: Vec<usize> = order
            .into_iter()
            .filter(|x| circuit_used.contains(x))
            .collect();
        // each element points to a variable or an instruction
        // if some element is dependent on another element, there is an edge between them
        let order_map = StaticHashMap::new(&order);
        let mut vertices_start_id = Vec::new();
        let mut num_vertices = 0;
        for circuit_id in order.iter() {
            let circuit = self.circuits.get(circuit_id).unwrap();
            let cur_vertices_nums = [
                circuit.get_num_variables() + 1,
                circuit.instructions.len(),
                circuit.outputs.len(),
                1,
            ];
            let mut t: [usize; 4] = [0; 4];
            for (i, cur_vertices_num) in cur_vertices_nums.iter().enumerate() {
                t[i] = num_vertices;
                num_vertices += cur_vertices_num;
            }
            vertices_start_id.push(t);
        }
        let vertice_id =
            |x: &Element| vertices_start_id[order_map.get(x.circuit_id)][x.typ as usize] + x.id;
        let mut edges_raw: Vec<(usize, usize)> = Vec::new();
        let mut dual_edges_raw: Vec<(usize, usize, usize)> = Vec::new();
        let mut add_edge = |from: Element, to: Element| {
            edges_raw.push((vertice_id(&from), vertice_id(&to)));
        };
        let mut add_dual_edge = |req1: Element, req2: Element, to: Element| {
            dual_edges_raw.push((vertice_id(&req1), vertice_id(&req2), vertice_id(&to)));
            dual_edges_raw.push((vertice_id(&req2), vertice_id(&req1), vertice_id(&to)));
        };
        for circuit_id in order.iter() {
            let circuit = self.circuits.get(circuit_id).unwrap();
            let mut cur_var_max = circuit.get_num_inputs_all();
            for (i, insn) in circuit.instructions.iter().enumerate() {
                if let Some((sub_circuit_id, inputs, num_outputs)) = insn.as_sub_circuit_call() {
                    for j in 0..num_outputs {
                        cur_var_max += 1;
                        add_edge(
                            Element {
                                circuit_id: *circuit_id,
                                id: cur_var_max,
                                typ: Var,
                            },
                            Element {
                                circuit_id: sub_circuit_id,
                                id: j,
                                typ: Output,
                            },
                        );
                        add_edge(
                            Element {
                                circuit_id: *circuit_id,
                                id: cur_var_max,
                                typ: Var,
                            },
                            Element {
                                circuit_id: *circuit_id,
                                id: i,
                                typ: Insn,
                            },
                        )
                    }
                    for (j, input) in inputs.iter().enumerate() {
                        add_dual_edge(
                            Element {
                                circuit_id: sub_circuit_id,
                                id: j + 1,
                                typ: Var,
                            },
                            Element {
                                circuit_id: *circuit_id,
                                id: i,
                                typ: Insn,
                            },
                            Element {
                                circuit_id: *circuit_id,
                                id: *input,
                                typ: Var,
                            },
                        );
                    }
                    add_edge(
                        Element {
                            circuit_id: sub_circuit_id,
                            id: 0,
                            typ: CircuitConstrained,
                        },
                        Element {
                            circuit_id: *circuit_id,
                            id: i,
                            typ: Insn,
                        },
                    );
                    add_edge(
                        Element {
                            circuit_id: sub_circuit_id,
                            id: 0,
                            typ: CircuitConstrained,
                        },
                        Element {
                            circuit_id: *circuit_id,
                            id: 0,
                            typ: CircuitConstrained,
                        },
                    );
                } else {
                    for _ in 0..insn.num_outputs() {
                        cur_var_max += 1;
                        add_edge(
                            Element {
                                circuit_id: *circuit_id,
                                id: cur_var_max,
                                typ: Var,
                            },
                            Element {
                                circuit_id: *circuit_id,
                                id: i,
                                typ: Insn,
                            },
                        );
                    }
                    for input in insn.inputs().iter() {
                        add_edge(
                            Element {
                                circuit_id: *circuit_id,
                                id: i,
                                typ: Insn,
                            },
                            Element {
                                circuit_id: *circuit_id,
                                id: *input,
                                typ: Var,
                            },
                        );
                    }
                }
            }
            assert_eq!(cur_var_max, circuit.get_num_variables());
            for (i, out) in circuit.outputs.iter().enumerate() {
                add_edge(
                    Element {
                        circuit_id: *circuit_id,
                        id: i,
                        typ: Output,
                    },
                    Element {
                        circuit_id: *circuit_id,
                        id: *out,
                        typ: Var,
                    },
                )
            }
        }
        // now we build the actual graph from raw edges
        let (edges_sorted_y, _) = bucket_sort(edges_raw, |x| x.1, num_vertices);
        let (edges, mut edges_start_pos) = bucket_sort(edges_sorted_y, |x| x.0, num_vertices);
        edges_start_pos.push(edges.len());
        let (dual_edges_sorted_z, _) = bucket_sort(dual_edges_raw, |x| x.2, num_vertices);
        let (dual_edges_sorted_y, _) = bucket_sort(dual_edges_sorted_z, |x| x.1, num_vertices);
        let (dual_edges, mut dual_edges_start_pos) =
            bucket_sort(dual_edges_sorted_y, |x| x.0, num_vertices);
        dual_edges_start_pos.push(dual_edges.len());
        // a queue is used to traverse the graph, and constraints are added first
        let mut queue: Vec<usize> = Vec::new();
        for circuit_id in order.iter() {
            let circuit = self.circuits.get(circuit_id).unwrap();
            for con in circuit.constraints.iter() {
                queue.push(vertice_id(&Element {
                    circuit_id: *circuit_id,
                    id: con.var(),
                    typ: Var,
                }));
            }
            if !circuit.constraints.is_empty() {
                queue.push(vertice_id(&Element {
                    circuit_id: *circuit_id,
                    id: 0,
                    typ: CircuitConstrained,
                }));
            }
        }
        for i in 0..self.circuits[&0].outputs.len() {
            queue.push(vertice_id(&Element {
                circuit_id: 0,
                id: i,
                typ: Output,
            }));
        }
        // after the traversal, the reachable elements are in the visited set
        let mut visited: Vec<bool> = vec![false; num_vertices];
        for &x in queue.iter() {
            visited[x] = true;
        }
        let mut i = 0;
        while i < queue.len() {
            let cur = queue[i];
            i += 1;
            for (_, neighbor) in edges
                .iter()
                .take(edges_start_pos[cur + 1])
                .skip(edges_start_pos[cur])
            {
                if !visited[*neighbor] {
                    visited[*neighbor] = true;
                    queue.push(*neighbor);
                }
            }
            for (_, req2, neighbor) in dual_edges
                .iter()
                .take(dual_edges_start_pos[cur + 1])
                .skip(dual_edges_start_pos[cur])
            {
                if visited[*req2] && !visited[*neighbor] {
                    visited[*neighbor] = true;
                    queue.push(*neighbor);
                }
            }
        }
        let is_visited = |x: &Element| visited[vertice_id(x)];
        // now we can process each circuit
        let mut new_circuits: HashMap<usize, Circuit<Irc>> = HashMap::new();
        for circuit_id in order.iter().rev() {
            let circuit = self.circuits.get(circuit_id).unwrap();
            let mut var_map: Vec<usize> = vec![EMPTY];
            let mut new_instructions: Vec<Irc::Instruction> = Vec::new();
            let mut mapped_var_max: usize = 0;
            for i in 1..=circuit.get_num_inputs_all() {
                if is_visited(&Element {
                    circuit_id: *circuit_id,
                    id: i,
                    typ: Var,
                }) {
                    mapped_var_max += 1;
                    var_map.push(mapped_var_max);
                } else {
                    var_map.push(EMPTY);
                }
            }
            for (i, insn) in circuit.instructions.iter().enumerate() {
                if !is_visited(&Element {
                    circuit_id: *circuit_id,
                    id: i,
                    typ: Insn,
                }) {
                    var_map.resize(var_map.len() + insn.num_outputs(), EMPTY);
                    continue;
                }
                if let Some((sub_circuit_id, inputs, num_outputs)) = insn.as_sub_circuit_call() {
                    let mut new_inputs: Vec<usize> = Vec::new();
                    let mut new_num_outputs: usize = 0;
                    for j in 0..inputs.len() {
                        if is_visited(&Element {
                            circuit_id: sub_circuit_id,
                            id: j + 1,
                            typ: Var,
                        }) {
                            new_inputs.push(var_map[inputs[j]]);
                        }
                    }
                    for j in 0..num_outputs {
                        if is_visited(&Element {
                            circuit_id: sub_circuit_id,
                            id: j,
                            typ: Output,
                        }) {
                            mapped_var_max += 1;
                            var_map.push(mapped_var_max);
                            new_num_outputs += 1;
                        } else {
                            var_map.push(EMPTY);
                        }
                    }
                    new_instructions.push(Irc::Instruction::sub_circuit_call(
                        sub_circuit_id,
                        new_inputs,
                        new_num_outputs,
                    ));
                } else {
                    let new_insn = insn.replace_vars(|x| var_map[x]);
                    new_instructions.push(new_insn);
                    for _ in 0..insn.num_outputs() {
                        mapped_var_max += 1;
                        var_map.push(mapped_var_max);
                    }
                }
            }
            assert_eq!(var_map.len(), circuit.get_num_variables() + 1);
            let mut new_outputs: Vec<usize> = Vec::new();
            for (i, out) in circuit.outputs.iter().enumerate() {
                if is_visited(&Element {
                    circuit_id: *circuit_id,
                    id: i,
                    typ: Output,
                }) {
                    new_outputs.push(var_map[*out]);
                }
            }
            new_circuits.insert(
                *circuit_id,
                Circuit {
                    instructions: new_instructions,
                    constraints: circuit
                        .constraints
                        .iter()
                        .map(|x| x.replace_var(|y| var_map[y]))
                        .collect(),
                    outputs: new_outputs,
                    num_inputs: (1..=circuit.num_inputs)
                        .filter(|x| var_map[*x] != EMPTY)
                        .count(),
                },
            );
        }
        let mut root_input_mask: Vec<bool> = Vec::new();
        for i in 1..=self.circuits[&0].num_inputs {
            root_input_mask.push(is_visited(&Element {
                circuit_id: 0,
                id: i,
                typ: Var,
            }));
        }
        let mut input_mapping_vec: Vec<usize> = Vec::new();
        let mut new_input_size: usize = 0;
        for v in root_input_mask.iter() {
            if *v {
                input_mapping_vec.push(new_input_size);
                new_input_size += 1;
            } else {
                input_mapping_vec.push(EMPTY);
            }
        }
        (
            RootCircuit {
                num_public_inputs: self.num_public_inputs,
                expected_num_output_zeroes: self.expected_num_output_zeroes,
                circuits: new_circuits,
            },
            InputMapping::new(new_input_size, input_mapping_vec),
        )
    }

    fn has_duplicate_sub_circuit_outputs(&self) -> bool {
        for circuit in self.circuits.values() {
            let out_set: HashSet<usize> = circuit.outputs.iter().cloned().collect();
            if out_set.len() != circuit.outputs.len() {
                return true;
            }
        }
        false
    }

    pub fn reassign_duplicate_sub_circuit_outputs(&mut self, force: bool) {
        if !self.has_duplicate_sub_circuit_outputs() {
            return;
        }
        let order = self.topo_order();
        let mut out_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for circuit_id in order.iter().rev() {
            let circuit = self.circuits.get_mut(circuit_id).unwrap();
            let mut var_map = Vec::with_capacity(circuit.get_num_variables() + 1);
            var_map.push(0);
            for i in 1..=circuit.get_num_inputs_all() {
                var_map.push(i);
            }
            for insn in circuit.instructions.iter_mut() {
                if let Some((sub_circuit_id, _, _)) = insn.as_sub_circuit_call() {
                    let t = var_map.len();
                    let om = out_map.get(&sub_circuit_id).unwrap();
                    for x in om.iter() {
                        var_map.push(*x + t);
                    }
                } else {
                    for _ in 0..insn.num_outputs() {
                        var_map.push(var_map.len());
                    }
                }
                *insn = insn.replace_vars(|x| var_map[x]);
            }
            for cons in circuit.constraints.iter_mut() {
                *cons = cons.replace_var(|x| var_map[x]);
            }
            for out in circuit.outputs.iter_mut() {
                *out = var_map[*out];
            }
            let mut om = Vec::with_capacity(circuit.outputs.len());
            for out in circuit.outputs.iter() {
                om.push(circuit.outputs.iter().position(|x| *x == *out).unwrap());
            }
            if *circuit_id == 0 || force {
                let mut var_max = var_map.len() - 1;
                for (i, x) in om.iter().enumerate() {
                    if *x < i {
                        circuit.instructions.push(Irc::Instruction::from_kx_plus_b(
                            circuit.outputs[*x],
                            CircuitField::<<Irc as IrConfig>::Config>::one(),
                            CircuitField::<<Irc as IrConfig>::Config>::zero(),
                        ));
                        var_max += 1;
                        circuit.outputs[i] = var_max;
                    }
                }
            }
            out_map.insert(*circuit_id, om);
        }
    }
}
