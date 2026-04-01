fn main() {
    #[cfg(feature = "asm")]
    if std::env::consts::ARCH != "x86_64" {
        eprintln!("Currently feature `asm` can only be enabled on x86_64 arch.");
        std::process::exit(1);
    }
    #[cfg(feature = "bn256-table")]
    {
        if std::path::Path::new("src/bn256/fr/table.rs").exists() {
            eprintln!("Pre-computed table for BN256 scalar field exists.");
            eprintln!("Skip pre-computation\n");
        } else {
            eprintln!("Generating pre-computed table for BN256 scalar field\n");
            std::process::Command::new("python3")
                .args(["script/bn256.py"])
                .output()
                .expect("requires python 3 to build pre-computed table");
        }
    }
}
