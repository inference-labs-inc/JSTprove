pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        let parent = (0..n).collect();
        Self { parent }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let x = self.find(x);
        let y = self.find(y);
        self.parent[x] = y;
    }
}
