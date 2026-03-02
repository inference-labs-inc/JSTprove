// credit: https://github.com/microsoft/Spartan/blob/master/src/timer.rs

#[cfg(feature = "profile")]
use colored::Colorize;
#[cfg(feature = "profile")]
use core::sync::atomic::AtomicUsize;
#[cfg(feature = "profile")]
use core::sync::atomic::Ordering;
#[cfg(feature = "profile")]
use std::time::Instant;

#[cfg(feature = "profile")]
pub static CALL_DEPTH: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "profile")]
pub struct Timer {
    label: String,
    timer: Instant,
    is_root: bool,
}

#[cfg(feature = "profile")]
impl Timer {
    #[inline(always)]
    pub fn new(label: &str, is_root: bool) -> Self {
        if !is_root {
            Self {
                label: label.to_string(),
                timer: Instant::now(),
                is_root: false,
            }
        } else {
            let timer = Instant::now();
            CALL_DEPTH.fetch_add(1, Ordering::Relaxed);
            let star = "* ";
            println!(
                "{:indent$}{}{}",
                "",
                star,
                label.yellow().bold(),
                indent = 2 * CALL_DEPTH.fetch_add(0, Ordering::Relaxed)
            );
            Self {
                label: label.to_string(),
                timer,
                is_root,
            }
        }
    }

    #[inline(always)]
    pub fn stop(&self) {
        if self.is_root {
            let duration = self.timer.elapsed();
            let star = "* ";
            println!(
                "{:indent$}{}{} {:?}",
                "",
                star,
                self.label.blue().bold(),
                duration,
                indent = 2 * CALL_DEPTH.fetch_add(0, Ordering::Relaxed)
            );
            CALL_DEPTH.fetch_sub(1, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn print(&self, msg: &str) {
        if self.is_root {
            CALL_DEPTH.fetch_add(1, Ordering::Relaxed);
            let star = "* ";
            println!(
                "{:indent$}{}{}",
                "",
                star,
                msg.to_string().green().bold(),
                indent = 2 * CALL_DEPTH.fetch_add(0, Ordering::Relaxed)
            );
            CALL_DEPTH.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[cfg(not(feature = "profile"))]
pub struct Timer {}

#[cfg(not(feature = "profile"))]
impl Timer {
    #[inline(always)]
    pub fn new(_label: &str, _is_root: bool) -> Self {
        Self {}
    }

    #[inline(always)]
    pub fn stop(&self) {}

    #[inline(always)]
    pub fn print(&self, _msg: &str) {}
}
