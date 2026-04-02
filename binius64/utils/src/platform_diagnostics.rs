// Copyright 2025 Irreducible Inc.
use std::{collections::BTreeMap, sync::OnceLock};

use regex::Regex;

// Build-time constants from environment variables
const BUILD_TARGET: &str = env!("BUILD_TARGET");
const BUILD_RUSTFLAGS: &str = env!("BUILD_RUSTFLAGS");
const COMPILE_TIME_FEATURES_STR: &str = env!("COMPILE_TIME_FEATURES");

// Lazy-initialized regex patterns for codebase scanning
static CFG_REGEX: OnceLock<Regex> = OnceLock::new();
static DETECT_REGEX: OnceLock<Regex> = OnceLock::new();

/// Creates a regex pattern that matches Rust `#[cfg(target_feature = "...")]` attributes.
///
/// This pattern is used to scan Rust source files and extract CPU features that are
/// conditionally compiled based on the target platform's capabilities.
///
/// # Pattern Details
/// - Matches: `target_feature = "feature_name"`
/// - Captures: The feature name (without quotes)
/// - Handles: Variable whitespace around `=`
///
/// # Example Matches
/// - `#[cfg(target_feature = "neon")]`
/// - `#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]`
fn cfg_feature_regex() -> &'static Regex {
    CFG_REGEX.get_or_init(|| {
        Regex::new(r#"target_feature\s*=\s*"([^"]+)""#)
            .expect("Failed to compile cfg feature regex")
    })
}

/// Creates a regex pattern that matches Rust feature detection macro calls.
///
/// This pattern is used to find runtime CPU feature detection in the codebase,
/// such as `is_x86_feature_detected!("avx")` or `is_aarch64_feature_detected!("neon")`.
///
/// # Pattern Details
/// - Matches: `is_*_feature_detected!("feature_name")`
/// - Captures: The feature name (without quotes)
/// - Handles: Any architecture prefix (x86, aarch64, etc.)
///
/// # Example Matches
/// - `is_x86_feature_detected!("avx2")`
/// - `is_aarch64_feature_detected!("neon")`
/// - `if is_x86_feature_detected!("gfni") { ... }`
fn runtime_detection_regex() -> &'static Regex {
    DETECT_REGEX.get_or_init(|| {
        Regex::new(r#"is_\w+_feature_detected!\s*\(\s*"([^"]+)"\s*\)"#)
            .expect("Failed to compile runtime detection regex")
    })
}

// Configuration constants
mod config {
    // CPU vendor detection strings
    pub const VENDOR_PATTERNS: &[(&str, &str)] = &[
        ("apple", "Apple"),
        ("graviton", "AWS"),
        ("ampere", "Ampere"),
        ("intel", "Intel"),
        ("amd", "AMD"),
    ];
    pub const VENDOR_GENERIC: &str = "Generic";

    // Architecture names (used for testing)
    #[cfg(test)]
    pub const KNOWN_ARCHITECTURES: &[&str] =
        &["x86_64", "aarch64", "arm", "riscv64", "wasm32", "wasm64"];

    // Operating systems (used for testing)
    #[cfg(test)]
    pub const KNOWN_OS: &[&str] = &[
        "linux", "macos", "windows", "freebsd", "openbsd", "netbsd", "android", "ios",
    ];

    // Features to categorize as SIMD (for display purposes)
    // Note: Features not in these lists will be categorized as "Other"
    pub const SIMD_FEATURES: &[&str] = &[
        "neon", "sve", "sve2", "dotprod", "fp16", "bf16", "i8mm", "f32mm", "f64mm", "fcma",
    ];

    // Features to categorize as Crypto (for display purposes)
    pub const CRYPTO_FEATURES: &[&str] = &["aes", "sha2", "sha3", "crc", "pmuv3"];

    // Display settings
    pub const MAX_DIRS_TO_SHOW: usize = 5;
    pub const MAX_FILES_IN_DIR: usize = 10;

    // Default values
    pub const UNKNOWN_CPU: &str = "Unknown CPU";
    pub const UNKNOWN_VERSION: &str = "unknown";

    // Directory names to skip
    pub const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules"];
    pub const RUST_FILE_EXT: &str = "rs";
    pub const ARCH_DIR_NAME: &str = "arch";

    // CPU target strategies
    pub const CPU_TARGET_NATIVE: &str = "native";
    pub const CPU_TARGET_GENERIC: &str = "generic";
}

pub struct PlatformDiagnostics {
    hardware: HardwareInfo,
    os_runtime: OSRuntimeInfo,
    llvm_config: LLVMConfig,
    code_features: CodeFeatures,
    codebase_usage: CodebaseUsage,
}

#[derive(Debug)]
struct HardwareInfo {
    cpu_model: String,
    architecture: &'static str,
    vendor: String,
    core_count: usize,
}

#[derive(Debug)]
struct OSRuntimeInfo {
    os: &'static str,
    kernel_version: String,
    runtime_features: BTreeMap<&'static str, bool>,
}

#[derive(Debug)]
struct LLVMConfig {
    target_triple: String,
    target_cpu: String,
}

#[derive(Debug)]
struct CodeFeatures {
    compile_time_features: Vec<String>,
    runtime_detected_features: BTreeMap<&'static str, bool>,
}

#[derive(Debug)]
struct CodebaseUsage {
    cfg_features: BTreeMap<String, Vec<String>>, // feature -> files using it
    runtime_detections: BTreeMap<String, Vec<String>>, // feature -> files using runtime detection
    arch_modules: Vec<String>,                   // architecture-specific modules found
}

// ANSI color codes
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

// Platform detection helper functions
#[cfg(target_os = "macos")]
fn get_macos_cpu_brand() -> Option<String> {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

#[cfg(target_os = "macos")]
fn get_kernel_version_via_uname() -> Option<String> {
    std::process::Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

// Runtime feature detection functions
#[cfg(target_arch = "aarch64")]
fn detect_aarch64_features() -> BTreeMap<&'static str, bool> {
    use std::arch::is_aarch64_feature_detected;
    let mut features = BTreeMap::new();

    // Note: We can't use a loop here because the macro requires literal strings
    features.insert("neon", is_aarch64_feature_detected!("neon"));
    features.insert("aes", is_aarch64_feature_detected!("aes"));
    features.insert("sha2", is_aarch64_feature_detected!("sha2"));
    features.insert("sha3", is_aarch64_feature_detected!("sha3"));
    features.insert("crc", is_aarch64_feature_detected!("crc"));
    features.insert("lse", is_aarch64_feature_detected!("lse"));
    features.insert("dotprod", is_aarch64_feature_detected!("dotprod"));
    features.insert("fp16", is_aarch64_feature_detected!("fp16"));
    features.insert("sve", is_aarch64_feature_detected!("sve"));
    features.insert("sve2", is_aarch64_feature_detected!("sve2"));
    features.insert("fcma", is_aarch64_feature_detected!("fcma"));
    features.insert("rcpc", is_aarch64_feature_detected!("rcpc"));
    features.insert("rcpc2", is_aarch64_feature_detected!("rcpc2"));
    features.insert("dpb", is_aarch64_feature_detected!("dpb"));
    features.insert("dpb2", is_aarch64_feature_detected!("dpb2"));
    features.insert("bf16", is_aarch64_feature_detected!("bf16"));
    features.insert("i8mm", is_aarch64_feature_detected!("i8mm"));
    features.insert("f32mm", is_aarch64_feature_detected!("f32mm"));
    features.insert("f64mm", is_aarch64_feature_detected!("f64mm"));

    features
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_64_features() -> BTreeMap<&'static str, bool> {
    use std::arch::is_x86_feature_detected;
    let mut features = BTreeMap::new();

    // Note: We can't use a loop here because the macro requires literal strings
    features.insert("avx", is_x86_feature_detected!("avx"));
    features.insert("avx2", is_x86_feature_detected!("avx2"));
    features.insert("avx512f", is_x86_feature_detected!("avx512f"));
    features.insert("gfni", is_x86_feature_detected!("gfni"));
    features.insert("aes", is_x86_feature_detected!("aes"));
    features.insert("pclmulqdq", is_x86_feature_detected!("pclmulqdq"));
    features.insert("sha", is_x86_feature_detected!("sha"));
    features.insert("vaes", is_x86_feature_detected!("vaes"));
    features.insert("vpclmulqdq", is_x86_feature_detected!("vpclmulqdq"));

    features
}

impl PlatformDiagnostics {
    #[must_use]
    pub fn gather() -> Self {
        Self {
            hardware: Self::detect_hardware(),
            os_runtime: Self::detect_os_runtime(),
            llvm_config: Self::parse_llvm_config(),
            code_features: Self::analyze_code_features(),
            codebase_usage: Self::scan_codebase_usage(),
        }
    }

    fn detect_hardware() -> HardwareInfo {
        let cpu_model = Self::get_cpu_model();
        let vendor = Self::detect_vendor(&cpu_model);
        let core_count = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);

        HardwareInfo {
            cpu_model,
            architecture: std::env::consts::ARCH,
            vendor,
            core_count,
        }
    }

    fn detect_vendor(cpu_model: &str) -> String {
        let model_lower = cpu_model.to_lowercase();
        for (pattern, vendor) in config::VENDOR_PATTERNS {
            if model_lower.contains(pattern) {
                return vendor.to_string();
            }
        }
        config::VENDOR_GENERIC.to_string()
    }

    fn get_cpu_model() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                // For x86_64
                if let Some(line) = cpuinfo.lines().find(|l| l.starts_with("model name")) {
                    return line.split(':').nth(1).unwrap_or("").trim().to_string();
                }
                // For ARM
                if let Some(line) = cpuinfo.lines().find(|l| l.starts_with("CPU implementer")) {
                    let implementer = line.split(':').nth(1).unwrap_or("").trim();
                    if let Some(part_line) = cpuinfo.lines().find(|l| l.starts_with("CPU part")) {
                        let part = part_line.split(':').nth(1).unwrap_or("").trim();
                        return format!("ARM implementer {implementer} part {part}");
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Some(cpu_brand) = get_macos_cpu_brand() {
                return cpu_brand;
            }
        }

        config::UNKNOWN_CPU.to_string()
    }

    fn detect_os_runtime() -> OSRuntimeInfo {
        #[cfg(target_arch = "aarch64")]
        let runtime_features = detect_aarch64_features();

        #[cfg(target_arch = "x86_64")]
        let runtime_features = detect_x86_64_features();

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let runtime_features = BTreeMap::new();

        let kernel_version = Self::get_kernel_version();

        OSRuntimeInfo {
            os: std::env::consts::OS,
            kernel_version,
            runtime_features,
        }
    }

    fn get_kernel_version() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/version")
                .ok()
                .and_then(|s| s.split_whitespace().nth(2).map(|s| s.to_string()))
                .unwrap_or_else(|| config::UNKNOWN_VERSION.to_string())
        }

        #[cfg(target_os = "macos")]
        {
            get_kernel_version_via_uname().unwrap_or_else(|| config::UNKNOWN_VERSION.to_string())
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            config::UNKNOWN_VERSION.to_string()
        }
    }

    fn parse_llvm_config() -> LLVMConfig {
        // Parse target-cpu from RUSTFLAGS
        // Handles both "-C target-cpu=native" and "-Ctarget-cpu=native" formats
        let mut target_cpu = config::CPU_TARGET_GENERIC.to_string();

        // Try to find target-cpu in RUSTFLAGS
        for (i, part) in BUILD_RUSTFLAGS.split_whitespace().enumerate() {
            if part == "-C" {
                // Check next part for "target-cpu=value"
                if let Some(next) = BUILD_RUSTFLAGS.split_whitespace().nth(i + 1)
                    && let Some(cpu) = next.strip_prefix("target-cpu=")
                {
                    target_cpu = cpu.to_string();
                    break;
                }
            } else if let Some(rest) = part.strip_prefix("-C") {
                // Handle "-Ctarget-cpu=value" (no space)
                if let Some(cpu) = rest.strip_prefix("target-cpu=") {
                    target_cpu = cpu.to_string();
                    break;
                }
            }
        }

        LLVMConfig {
            target_triple: BUILD_TARGET.to_string(),
            target_cpu,
        }
    }

    fn analyze_code_features() -> CodeFeatures {
        let compile_time_features = COMPILE_TIME_FEATURES_STR
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        let runtime_detected_features = Self::detect_os_runtime().runtime_features;

        CodeFeatures {
            compile_time_features,
            runtime_detected_features,
        }
    }

    fn scan_codebase_usage() -> CodebaseUsage {
        let mut cfg_features = BTreeMap::new();
        let mut runtime_detections = BTreeMap::new();
        let mut arch_modules = Vec::new();

        // Try to find the workspace root
        let workspace_root = std::env::var("CARGO_MANIFEST_DIR").ok().and_then(|dir| {
            let path = std::path::Path::new(&dir);
            // Walk up to find workspace root (has Cargo.toml with [workspace])
            let mut current = Some(path);
            while let Some(p) = current {
                let cargo_toml = p.join("Cargo.toml");
                if cargo_toml.exists()
                    && let Ok(content) = std::fs::read_to_string(&cargo_toml)
                    && content.contains("[workspace]")
                {
                    return p.to_str().map(std::string::ToString::to_string);
                }
                current = p.parent();
            }
            None
        });

        if let Some(root) = workspace_root {
            // Get regex patterns for scanning
            let cfg_regex = cfg_feature_regex();
            let detect_regex = runtime_detection_regex();

            // Scan for arch modules and features
            Self::scan_directory_with_regex(
                std::path::Path::new(&root),
                &mut cfg_features,
                &mut runtime_detections,
                &mut arch_modules,
                &root,
                cfg_regex,
                detect_regex,
            );

            arch_modules.sort();
            arch_modules.dedup();
        }

        CodebaseUsage {
            cfg_features,
            runtime_detections,
            arch_modules,
        }
    }

    fn scan_directory_with_regex(
        dir: &std::path::Path,
        cfg_features: &mut BTreeMap<String, Vec<String>>,
        runtime_detections: &mut BTreeMap<String, Vec<String>>,
        arch_modules: &mut Vec<String>,
        root: &str,
        cfg_regex: &Regex,
        detect_regex: &Regex,
    ) {
        // Skip common non-source directories
        if let Some(name) = dir.file_name().and_then(|n| n.to_str())
            && (config::SKIP_DIRS.contains(&name) || name.starts_with('.'))
        {
            return;
        }

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.is_dir() {
                    // Check if this is an arch module
                    if path.file_name() == Some(std::ffi::OsStr::new(config::ARCH_DIR_NAME)) {
                        // List subdirectories as arch modules
                        if let Ok(arch_entries) = std::fs::read_dir(&path) {
                            for arch_entry in arch_entries.flatten() {
                                if arch_entry.path().is_dir()
                                    && let Some(name) = arch_entry.file_name().to_str()
                                {
                                    arch_modules.push(name.to_string());
                                }
                            }
                        }
                    }

                    // Recurse into subdirectory
                    Self::scan_directory_with_regex(
                        &path,
                        cfg_features,
                        runtime_detections,
                        arch_modules,
                        root,
                        cfg_regex,
                        detect_regex,
                    );
                } else if path.extension() == Some(std::ffi::OsStr::new(config::RUST_FILE_EXT)) {
                    // Scan Rust file for features
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        let relative_path = path
                            .strip_prefix(root)
                            .unwrap_or(&path)
                            .to_string_lossy()
                            .to_string();

                        // Find all cfg features
                        for cap in cfg_regex.captures_iter(&content) {
                            if let Some(feature_match) = cap.get(1) {
                                let feature = feature_match.as_str();
                                // Skip invalid feature names
                                if !feature.is_empty() && !feature.contains('.') {
                                    cfg_features
                                        .entry(feature.to_string())
                                        .or_default()
                                        .push(relative_path.clone());
                                }
                            }
                        }

                        // Find all runtime detections
                        for cap in detect_regex.captures_iter(&content) {
                            if let Some(feature_match) = cap.get(1) {
                                let feature = feature_match.as_str();
                                // Skip invalid feature names and generic placeholders
                                if !feature.is_empty()
                                    && !feature.contains('.')
                                    && feature != "feature"
                                {
                                    runtime_detections
                                        .entry(feature.to_string())
                                        .or_default()
                                        .push(relative_path.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn print(&self) {
        println!("\n{BOLD}Platform Feature Report{RESET}\n");

        // 1. Hardware
        self.print_hardware();
        println!();

        // 2. OS/Runtime
        self.print_os_runtime();
        println!();

        // 3. Compilation Target (LLVM)
        self.print_llvm();
        println!();

        // 4. Available CPU Instructions
        self.print_available_instructions();
        println!();

        // 5. Codebase Usage
        self.print_codebase_usage();
    }

    fn print_hardware(&self) {
        println!(
            "{BOLD}{CYAN}Hardware:{RESET} {} {} ({} cores)",
            self.hardware.vendor, self.hardware.architecture, self.hardware.core_count
        );
        println!("{CYAN}CPU:{RESET} {}", self.hardware.cpu_model);

        match self.hardware.vendor.as_str() {
            "Apple" => {
                println!(
                    "{CYAN}Features:{RESET} {GREEN}✓{RESET}AMX, {GREEN}✓{RESET}Neural Engine, {GREEN}✓{RESET}P+E cores, {RED}✗{RESET}SVE/SVE2, {GREEN}✓{RESET}NEON"
                );
            }
            "AWS" => {
                println!(
                    "{CYAN}Features:{RESET} {GREEN}✓{RESET}SVE-256bit, {GREEN}✓{RESET}Server memory, {GREEN}✓{RESET}Large cache, {RED}✗{RESET}AMX, {GREEN}✓{RESET}NEON"
                );
            }
            _ => {
                println!(
                    "{CYAN}Features:{RESET} {YELLOW}?{RESET}Vendor-specific, {GREEN}✓{RESET}NEON, {YELLOW}?{RESET}Crypto"
                );
            }
        }
    }

    fn print_os_runtime(&self) {
        println!(
            "{BOLD}{CYAN}OS/Runtime:{RESET} {} (kernel {})",
            self.os_runtime.os, self.os_runtime.kernel_version
        );

        // Group features by status
        let detected: Vec<&str> = self
            .os_runtime
            .runtime_features
            .iter()
            .filter(|(_, v)| **v)
            .map(|(k, _)| *k)
            .collect();
        let not_found: Vec<&str> = self
            .os_runtime
            .runtime_features
            .iter()
            .filter(|(_, v)| !**v)
            .map(|(k, _)| *k)
            .collect();

        if !detected.is_empty() {
            println!("{GREEN}Detected:{RESET} {}", detected.join(", "));
        }
        if !not_found.is_empty() {
            println!("{DIM}Not available:{RESET} {}", not_found.join(", "));
        }
    }

    fn print_llvm(&self) {
        println!("{BOLD}{CYAN}Compilation Target:{RESET}");
        println!("{CYAN}Triple:{RESET} {}", self.llvm_config.target_triple);
        println!("{CYAN}CPU:{RESET} {}", self.llvm_config.target_cpu);

        match self.llvm_config.target_cpu.as_str() {
            config::CPU_TARGET_NATIVE => {
                println!(
                    "{CYAN}Strategy:{RESET} {YELLOW}Native{RESET} - Optimized for this specific CPU"
                );
                println!("{DIM}         Binary only runs on CPUs with same features{RESET}");
            }
            config::CPU_TARGET_GENERIC => {
                println!(
                    "{CYAN}Strategy:{RESET} {GREEN}Generic{RESET} - Portable across all {} CPUs",
                    if self.llvm_config.target_triple.contains("aarch64") {
                        "ARM64"
                    } else if self.llvm_config.target_triple.contains("x86_64") {
                        "x86-64"
                    } else {
                        "target"
                    }
                );
                println!(
                    "{DIM}         Uses explicit features but no CPU-specific scheduling{RESET}"
                );
            }
            cpu if cpu.starts_with("apple-") => {
                println!(
                    "{CYAN}Strategy:{RESET} {MAGENTA}Apple Silicon{RESET} - Optimized for {cpu}"
                );
                println!("{DIM}         Enables AMX, disables SVE{RESET}");
            }
            cpu if cpu.contains("neoverse") => {
                println!("{CYAN}Strategy:{RESET} {BLUE}Server ARM{RESET} - Optimized for {cpu}");
                println!("{DIM}         Enables SVE, optimized for cloud workloads{RESET}");
            }
            _ => {
                println!("{CYAN}Strategy:{RESET} Custom CPU target");
            }
        }
    }

    fn print_available_instructions(&self) {
        println!("{BOLD}{CYAN}Available CPU Instructions:{RESET}");

        // Group features by category
        let mut simd_features = Vec::new();
        let mut crypto_features = Vec::new();
        let mut arch_features = Vec::new();

        for feature in &self.code_features.compile_time_features {
            if config::SIMD_FEATURES.contains(&feature.as_str()) {
                simd_features.push(feature.as_str());
            } else if config::CRYPTO_FEATURES.contains(&feature.as_str()) {
                crypto_features.push(feature.as_str());
            } else if !feature.starts_with("v8.") && feature != "vh" {
                arch_features.push(feature.as_str());
            }
        }

        println!(
            "{CYAN}Total:{RESET} {} CPU features available to compiler",
            self.code_features.compile_time_features.len()
        );

        if !simd_features.is_empty() {
            simd_features.sort_unstable();
            println!("  {GREEN}SIMD:{RESET} {}", simd_features.join(", "));
        }
        if !crypto_features.is_empty() {
            crypto_features.sort_unstable();
            println!("  {GREEN}Crypto:{RESET} {}", crypto_features.join(", "));
        }
        if !arch_features.is_empty() {
            arch_features.sort_unstable();
            // Always show the features, but wrap if too many
            if arch_features.len() <= 6 {
                println!("  {GREEN}Other:{RESET} {}", arch_features.join(", "));
            } else {
                // Show in multiple lines for readability
                println!("  {GREEN}Other:{RESET}");
                for chunk in arch_features.chunks(8) {
                    println!("    {}", chunk.join(", "));
                }
            }
        }

        // Show important missing features
        #[cfg(target_arch = "aarch64")]
        {
            let important_missing = vec!["sve", "sve2"]
                .into_iter()
                .filter(|f| {
                    !self
                        .code_features
                        .compile_time_features
                        .iter()
                        .any(|feature| feature == f)
                })
                .collect::<Vec<_>>();
            if !important_missing.is_empty() {
                println!(
                    "  {DIM}Not available:{RESET} {} (code paths requiring these are excluded)",
                    important_missing.join(", ")
                );
            }
        }
    }

    fn print_feature_locations(&self, _feature: &str, locations: &[String]) {
        // Group files by directory
        let mut by_dir: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for loc in locations {
            if let Some(slash_pos) = loc.rfind('/') {
                let dir = loc[..slash_pos].to_string();
                let file = loc[slash_pos + 1..].to_string();
                let files = by_dir.entry(dir).or_default();
                if !files.contains(&file) {
                    files.push(file);
                }
            } else {
                let files = by_dir.entry(String::new()).or_default();
                if !files.contains(loc) {
                    files.push(loc.clone());
                }
            }
        }

        let mut shown = 0;
        for (dir_count, (dir, files)) in by_dir.iter().enumerate() {
            if dir_count >= config::MAX_DIRS_TO_SHOW && by_dir.len() > config::MAX_DIRS_TO_SHOW {
                println!("    ... in {} more files", locations.len() - shown);
                break;
            }

            if files.len() == 1 {
                println!("    {}/{}", dir, files[0]);
                shown += 1;
            } else if files.len() <= config::MAX_FILES_IN_DIR {
                // List all files if 10 or fewer
                println!("    {}/: {}", dir, files.join(", "));
                shown += files.len();
            } else {
                // Show first 10 files and indicate there are more
                let first_10: Vec<_> = files
                    .iter()
                    .take(config::MAX_FILES_IN_DIR)
                    .cloned()
                    .collect();
                println!(
                    "    {}/: {} (and {} more)",
                    dir,
                    first_10.join(", "),
                    files.len() - config::MAX_FILES_IN_DIR
                );
                shown += files.len();
            }
        }
    }

    fn print_codebase_usage(&self) {
        // Always show the codebase section header
        println!("{BOLD}{CYAN}Codebase Analysis:{RESET}");

        if self.codebase_usage.cfg_features.is_empty()
            && self.codebase_usage.runtime_detections.is_empty()
            && self.codebase_usage.arch_modules.is_empty()
        {
            println!("{DIM}  No feature usage detected{RESET}");
            return;
        }

        // Show arch modules first
        if !self.codebase_usage.arch_modules.is_empty() {
            println!(
                "{MAGENTA}Arch modules:{RESET} {}",
                self.codebase_usage.arch_modules.join(", ")
            );
        }

        if !self.codebase_usage.cfg_features.is_empty() {
            // Check which used features are enabled vs disabled
            let mut enabled_used = Vec::new();
            let mut disabled_used = Vec::new();

            for feature in self.codebase_usage.cfg_features.keys() {
                // Check if feature is enabled at compile time
                if self.code_features.compile_time_features.contains(feature) {
                    enabled_used.push(feature.clone());
                } else {
                    disabled_used.push(feature.clone());
                }
            }

            if !enabled_used.is_empty() {
                println!("{GREEN}Used & Enabled:{RESET}");
                for feature in &enabled_used {
                    if let Some(locations) = self.codebase_usage.cfg_features.get(feature) {
                        println!("  {GREEN}{feature}:{RESET}");
                        self.print_feature_locations(feature, locations);
                    }
                }
            }

            if !disabled_used.is_empty() {
                println!("{DIM}Used but NOT Enabled:{RESET}");
                for feature in &disabled_used {
                    if let Some(locations) = self.codebase_usage.cfg_features.get(feature) {
                        println!("  {DIM}{feature}:{RESET}");
                        self.print_feature_locations(feature, locations);
                    }
                }
            }
        }

        if !self.codebase_usage.runtime_detections.is_empty() {
            let detections: Vec<String> = self
                .codebase_usage
                .runtime_detections
                .keys()
                .cloned()
                .collect();
            println!("{BLUE}Runtime detections:{RESET} {}", detections.join(", "));
        }
    }

    #[must_use]
    pub fn get_summary(&self) -> PlatformSummary {
        let has_mismatches =
            self.code_features.compile_time_features.iter().any(|f| {
                self.code_features.runtime_detected_features.get(f.as_str()) == Some(&false)
            });

        PlatformSummary {
            platform: format!("{} on {}", self.hardware.vendor, self.hardware.architecture),
            cpu: self.hardware.cpu_model.clone(),
            target: self.llvm_config.target_triple.clone(),
            target_cpu: self.llvm_config.target_cpu.clone(),
            has_feature_mismatches: has_mismatches,
        }
    }

    /// Generate a feature suffix for benchmark names based on platform diagnostics
    #[must_use]
    pub fn get_feature_suffix(&self) -> String {
        let mut suffix_parts = Vec::new();

        // Threading - check if rayon feature is enabled
        #[cfg(feature = "rayon")]
        suffix_parts.push("mt");
        #[cfg(not(feature = "rayon"))]
        suffix_parts.push("st");

        // Architecture
        #[cfg(target_arch = "x86_64")]
        {
            suffix_parts.push("x86");
            // Add key features based on compile-time features
            #[cfg(target_feature = "gfni")]
            suffix_parts.push("gfni");
            #[cfg(target_feature = "avx512f")]
            suffix_parts.push("avx512");
            #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
            suffix_parts.push("avx2");
        }

        #[cfg(target_arch = "aarch64")]
        {
            suffix_parts.push("arm64");
            // Check for NEON and AES
            #[cfg(all(target_feature = "neon", target_feature = "aes"))]
            suffix_parts.push("neon_aes");
            #[cfg(all(target_feature = "neon", not(target_feature = "aes")))]
            suffix_parts.push("neon");
        }

        suffix_parts.join("_")
    }
}

pub struct PlatformSummary {
    pub platform: String,
    pub cpu: String,
    pub target: String,
    pub target_cpu: String,
    pub has_feature_mismatches: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_diagnostics() {
        let diag = PlatformDiagnostics::gather();
        diag.print();

        // Also test the summary
        let summary = diag.get_summary();
        assert!(!summary.platform.is_empty());
        assert!(!summary.cpu.is_empty());
        assert!(!summary.target.is_empty());
    }

    #[test]
    fn test_sanity() {
        // Test that PlatformDiagnostics can be created without panicking
        let diag = PlatformDiagnostics::gather();

        // Test hardware info
        assert!(
            !diag.hardware.cpu_model.is_empty(),
            "CPU model should not be empty"
        );
        assert!(
            !diag.hardware.vendor.is_empty(),
            "Vendor should not be empty"
        );
        assert!(diag.hardware.core_count >= 1, "Should have at least 1 core");
        assert!(
            config::KNOWN_ARCHITECTURES.contains(&diag.hardware.architecture),
            "Architecture should be a known value"
        );

        // Test OS runtime info
        assert!(
            !diag.os_runtime.kernel_version.is_empty(),
            "Kernel version should not be empty"
        );
        assert!(
            config::KNOWN_OS.contains(&diag.os_runtime.os),
            "OS should be a known value"
        );

        // Test LLVM config
        assert!(
            !diag.llvm_config.target_triple.is_empty(),
            "Target triple should not be empty"
        );
        assert!(
            !diag.llvm_config.target_cpu.is_empty(),
            "Target CPU should not be empty"
        );

        // Test code features
        // Compile-time features can be empty on some platforms
        assert!(
            diag.code_features.compile_time_features.is_empty()
                || diag
                    .code_features
                    .compile_time_features
                    .iter()
                    .all(|f| !f.is_empty()),
            "All feature names should be non-empty"
        );

        // Test summary generation
        let summary = diag.get_summary();
        assert!(
            !summary.platform.is_empty(),
            "Summary platform should not be empty"
        );
        assert!(!summary.cpu.is_empty(), "Summary CPU should not be empty");
        assert!(
            !summary.target.is_empty(),
            "Summary target should not be empty"
        );
        assert!(
            !summary.target_cpu.is_empty(),
            "Summary target CPU should not be empty"
        );

        // Test that print() doesn't panic
        // Redirect output to avoid cluttering test output
        let _output = std::panic::catch_unwind(|| {
            diag.print();
        });
        assert!(_output.is_ok(), "print() should not panic");
    }

    #[test]
    fn test_detect_vendor() {
        assert_eq!(PlatformDiagnostics::detect_vendor("Apple M1 Pro"), "Apple");
        assert_eq!(PlatformDiagnostics::detect_vendor("Intel Core i7"), "Intel");
        assert_eq!(PlatformDiagnostics::detect_vendor("AMD Ryzen 9"), "AMD");
        assert_eq!(PlatformDiagnostics::detect_vendor("AWS Graviton3"), "AWS");
        assert_eq!(PlatformDiagnostics::detect_vendor("Ampere Altra"), "Ampere");
        assert_eq!(PlatformDiagnostics::detect_vendor("Unknown CPU"), "Generic");
    }
}
