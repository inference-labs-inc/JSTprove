fn main() {
    #[cfg(feature = "asm")]
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() != "x86_64" {
        eprintln!("Feature `asm` is only supported on x86_64 targets; ignoring on this target.");
    }
}
