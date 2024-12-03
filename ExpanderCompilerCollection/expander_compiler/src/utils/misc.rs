use std::collections::{HashMap, HashSet};

pub fn next_power_of_two(x: usize) -> usize {
    let mut padk: usize = 0;
    while (1 << padk) < x {
        padk += 1;
    }
    1 << padk
}

pub fn topo_order_and_is_dag(
    vertices: &HashSet<usize>,
    edges: &HashMap<usize, HashSet<usize>>,
) -> (Vec<usize>, bool) {
    let queue: Vec<usize> = topo_order(vertices, edges);
    let is_dag = queue.len() == vertices.len();
    (queue, is_dag)
}

// must be a DAG
pub fn topo_order(vertices: &HashSet<usize>, edges: &HashMap<usize, HashSet<usize>>) -> Vec<usize> {
    let mut queue: Vec<usize> = Vec::new();
    let mut in_deg: HashMap<usize, usize> = HashMap::new();
    for &from in vertices.iter() {
        in_deg.insert(from, 0);
    }
    for tos in edges.values() {
        for &to in tos.iter() {
            in_deg.entry(to).and_modify(|e| *e += 1);
        }
    }
    for from in vertices.iter() {
        if in_deg[from] == 0 {
            queue.push(*from);
        }
    }
    let mut i = 0;
    while i < queue.len() {
        let from = queue[i];
        i += 1;
        if let Some(tos) = edges.get(&from) {
            for &to in tos.iter() {
                in_deg.entry(to).and_modify(|e| *e -= 1);
                if in_deg[&to] == 0 {
                    queue.push(to);
                }
            }
        }
    }
    queue
}
