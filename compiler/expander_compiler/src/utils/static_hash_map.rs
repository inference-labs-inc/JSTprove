use rand::RngCore;

pub struct StaticHashMap {
    m: u64,
    a: u64,
    b: u64,
    v: Vec<usize>,
}

const MOD: u64 = 1_000_000_007;

impl StaticHashMap {
    pub fn new(s: &[usize]) -> Self {
        if s.len() > (MOD / 1000) as usize {
            panic!("too large");
        }
        let mut rng = rand::thread_rng();
        let mut m = 1;
        while m < (s.len() * 2) as u64 {
            m *= 2
        }
        loop {
            for _ in 0..10 {
                let a = rng.next_u64() % (MOD - 1) + 1;
                let b = rng.next_u64() % (MOD - 1) + 1;
                let mut v = vec![0; m as usize];
                let mut ok = true;
                for (i, &x) in s.iter().enumerate() {
                    let x = (x as u64) % MOD;
                    let pos = ((x * a + b) % MOD * x % MOD) & (m - 1);
                    if v[pos as usize] != 0 {
                        ok = false;
                        break;
                    }
                    v[pos as usize] = i + 1;
                }
                if ok {
                    for x in v.iter_mut() {
                        if *x > 0 {
                            *x -= 1;
                        }
                    }
                    return Self { m: m - 1, a, b, v };
                }
            }
            if m < MOD / 250 {
                m *= 2
            }
        }
    }

    pub fn get(&self, x: usize) -> usize {
        let x = (x as u64) % MOD;
        let pos = ((x * self.a + self.b) % MOD * x % MOD) & self.m;
        self.v[pos as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_hash_map() {
        let s = vec![5, 6, 7, 1, 2, 3, 8, 9, 10];
        let hm = StaticHashMap::new(&s);
        for (i, &x) in s.iter().enumerate() {
            assert_eq!(hm.get(x), i);
        }
    }
}
