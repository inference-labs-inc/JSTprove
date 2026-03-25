use std::time::Instant;

use console::style;
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
        eprintln!("      {}", style(msg).dim());
    }

    pub fn finish_ok(&self, msg: &str) {
        let elapsed = self.start.elapsed();
        let check = style("done").green().bold();
        let timing = style(format!("{:.2}s", elapsed.as_secs_f64())).dim();
        eprintln!("\n  {check} {msg} in {timing}");
    }

    pub fn finish_err(&self, msg: &str) {
        let x = style("fail").red().bold();
        eprintln!("\n  {x} {msg}");
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("      {spinner:.dim} {msg}")
            .expect("valid template")
            .tick_strings(&[".", "..", "...", ""]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(300));
    pb
}

pub fn progress_bar(len: u64, label: &str) -> ProgressBar {
    let escaped = label.replace('{', "{{").replace('}', "}}");
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "      {escaped} [{{bar:30.cyan/dim}}] {{pos}}/{{len}} {{msg}}"
        ))
        .expect("valid template")
        .progress_chars("== "),
    );
    pb
}

pub fn header(cmd: &str) {
    eprintln!("{} {}", style("jstprove").bold(), style(cmd).cyan().bold());
    eprintln!();
}

#[allow(clippy::cast_precision_loss)]
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
