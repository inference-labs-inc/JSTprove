#[cfg(feature = "profile")]
use std::thread;
#[cfg(feature = "profile")]
use utils::timer::Timer;

#[cfg(feature = "profile")]
#[test]
fn test_timer() {
    let block_1 = Timer::new("block 1", true);
    let block_2 = Timer::new("block 2", true);
    thread::sleep(std::time::Duration::from_secs(2));
    let block_3 = Timer::new("block 3", true);
    block_3.print("Some extra information from block 3");
    thread::sleep(std::time::Duration::from_secs(1));
    block_3.stop();
    block_2.stop();
    block_1.stop();
}
