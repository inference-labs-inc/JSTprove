use std::path::PathBuf;
use std::process::Command;

fn main() {
    prost_build::Config::new()
        .compile_protos(&["proto/onnx_ml.proto"], &["proto/"])
        .expect("prost protobuf compilation failed");

    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();

    let rev = match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    };

    println!("cargo:rustc-env=JSTPROVE_GIT_REV={rev}");

    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));

    let git_dir = manifest_dir
        .ancestors()
        .map(|p| p.join(".git"))
        .find(|g| g.exists());

    if let Some(git) = git_dir {
        let head = git.join("HEAD");
        if head.exists() {
            println!("cargo:rerun-if-changed={}", head.display());
        }
        let packed_refs = git.join("packed-refs");
        if packed_refs.exists() {
            println!("cargo:rerun-if-changed={}", packed_refs.display());
        }
        let refs_heads = git.join("refs").join("heads");
        if refs_heads.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&refs_heads) {
                for entry in entries.flatten() {
                    println!("cargo:rerun-if-changed={}", entry.path().display());
                }
            }
        }
    } else {
        println!("cargo:rerun-if-changed=.git/HEAD");
    }
}
