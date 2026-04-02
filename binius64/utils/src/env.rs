// Copyright 2024-2025 Irreducible Inc.

/// Read boolean flag from the environment variable.
pub fn boolean_env_flag_set(flag: &str) -> bool {
    match std::env::var(flag) {
        Ok(val) => ["1", "on", "ON", "true", "TRUE", "yes", "YES"].contains(&val.as_str()),
        Err(_) => false,
    }
}
