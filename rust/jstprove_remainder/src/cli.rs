use std::time::Instant;

use console::{style, Style};
use indicatif::{ProgressBar, ProgressStyle};

pub struct StepPrinter {
    step: usize,
    total: usize,
    start: Instant,
}

impl StepPrinter {
    pub fn new(total: usize) -> Self {
        Self {
            step: 0,
            total,
            start: Instant::now(),
        }
    }

    pub fn step(&mut self, msg: &str) {
        self.step += 1;
        let prefix = style(format!("[{}/{}]", self.step, self.total))
            .cyan()
            .bold();
        eprintln!("{prefix} {msg}");
    }

    pub fn detail(&self, msg: &str) {
        let indent = Style::new().dim();
        eprintln!("      {}", indent.apply_to(msg));
    }

    pub fn finish_ok(&self, msg: &str) {
        let elapsed = self.start.elapsed();
        let check = style("✓").green().bold();
        let timing = style(format!("({:.2}s)", elapsed.as_secs_f64())).dim();
        eprintln!("{check} {msg} {timing}");
    }

    pub fn finish_err(&self, msg: &str) {
        let x = style("✗").red().bold();
        eprintln!("{x} {msg}");
    }
}

pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

pub fn progress_bar(len: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "  {{spinner:.cyan}} {label} [{{bar:30.cyan/dim}}] {{pos}}/{{len}} {{msg}}"
        ))
        .unwrap()
        .progress_chars("━╸─")
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

pub fn fmt_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

pub fn header(cmd: &str) {
    let name = style("jstprove").bold();
    let cmd = style(cmd).cyan().bold();
    eprintln!("{name} {cmd}");
    eprintln!();
}
