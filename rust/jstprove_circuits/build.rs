use std::process::Command;

fn main() {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();

    let rev = match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    };

    println!("cargo:rustc-env=JSTPROVE_GIT_REV={rev}");
    println!("cargo:rerun-if-changed=.git/HEAD");
}
