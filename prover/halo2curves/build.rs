fn main() {
    #[cfg(feature = "asm")]
    if std::env::consts::ARCH != "x86_64" {
        eprintln!("Feature `asm` is only supported on x86_64; ignoring on this platform.");
    }
}
