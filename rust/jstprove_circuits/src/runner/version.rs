pub use jstprove_io::ArtifactVersion;

#[must_use]
pub fn jstprove_artifact_version() -> ArtifactVersion {
    jstprove_io::version::current()
}
