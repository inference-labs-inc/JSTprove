use std::any::TypeId;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn get_function_id<F: 'static>() -> u64 {
    let type_id = TypeId::of::<F>();
    let mut hasher = DefaultHasher::new();
    type_id.hash(&mut hasher);
    hasher.finish()
}
