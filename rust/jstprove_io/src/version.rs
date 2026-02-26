use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactVersion {
    pub crate_version: String,
    pub git_rev: String,
}

#[must_use]
pub fn current() -> ArtifactVersion {
    ArtifactVersion {
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        git_rev: env!("JSTPROVE_GIT_REV").to_string(),
    }
}
