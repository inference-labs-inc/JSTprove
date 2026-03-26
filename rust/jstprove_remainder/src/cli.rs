use std::time::Instant;

use console::style;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    Human,
    Quiet,
    Json,
}

pub struct StepPrinter {
    step: usize,
    total: usize,
    start: Instant,
    mode: OutputMode,
}

impl StepPrinter {
    pub fn new(total: usize, mode: OutputMode) -> Self {
        Self {
            step: 0,
            total,
            start: Instant::now(),
            mode,
        }
    }

    pub fn step(&mut self, msg: &str) {
        self.step += 1;
        match self.mode {
            OutputMode::Human => {
                let prefix = style(format!("[{}/{}]", self.step, self.total))
                    .cyan()
                    .bold();
                eprintln!("{prefix} {msg}");
            }
            OutputMode::Json => {
                let obj = serde_json::json!({
                    "event": "step",
                    "step": self.step,
                    "total": self.total,
                    "message": msg,
                });
                eprintln!("{obj}");
            }
            OutputMode::Quiet => {}
        }
    }

    pub fn detail(&self, msg: &str) {
        match self.mode {
            OutputMode::Human => {
                eprintln!("      {}", style(msg).dim());
            }
            OutputMode::Json => {
                let obj = serde_json::json!({
                    "event": "detail",
                    "step": self.step,
                    "message": msg,
                });
                eprintln!("{obj}");
            }
            OutputMode::Quiet => {}
        }
    }

    pub fn finish_ok(&self, msg: &str) {
        let elapsed = self.start.elapsed();
        match self.mode {
            OutputMode::Human => {
                let check = style("done").green().bold();
                let timing = style(format!("{:.2}s", elapsed.as_secs_f64())).dim();
                eprintln!("\n  {check} {msg} in {timing}");
            }
            OutputMode::Json => {
                let obj = serde_json::json!({
                    "event": "done",
                    "message": msg,
                    "elapsed_ms": elapsed.as_millis(),
                });
                eprintln!("{obj}");
            }
            OutputMode::Quiet => {}
        }
    }

    pub fn finish_err(&self, msg: &str) {
        match self.mode {
            OutputMode::Human => {
                let x = style("fail").red().bold();
                eprintln!("\n  {x} {msg}");
            }
            OutputMode::Json => {
                let elapsed = self.start.elapsed();
                let obj = serde_json::json!({
                    "event": "error",
                    "message": msg,
                    "elapsed_ms": elapsed.as_millis(),
                });
                eprintln!("{obj}");
            }
            OutputMode::Quiet => {}
        }
    }
}

pub fn spinner(msg: &str, mode: OutputMode) -> ProgressBar {
    if mode != OutputMode::Human {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("      {spinner:.dim} {msg}")
            .unwrap()
            .tick_strings(&[".", "..", "...", ""]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(300));
    pb
}

pub fn progress_bar(len: u64, label: &str, mode: OutputMode) -> ProgressBar {
    if mode != OutputMode::Human {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(&format!(
            "      {label} [{{bar:30.cyan/dim}}] {{pos}}/{{len}} {{msg}}"
        ))
        .unwrap()
        .progress_chars("== "),
    );
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

pub fn header(cmd: &str, mode: OutputMode) {
    match mode {
        OutputMode::Human => {
            eprintln!("{} {}", style("jstprove").bold(), style(cmd).cyan().bold());
            eprintln!();
        }
        OutputMode::Json => {
            let obj = serde_json::json!({
                "event": "start",
                "command": cmd,
            });
            eprintln!("{obj}");
        }
        OutputMode::Quiet => {}
    }
}
