
#!/bin/bash
cargo build --release --bin zkcuda_setup --bin zkcuda_prove --bin zkcuda_verify --bin zkcuda_cleanup --bin expander_server_no_oversubscribe

# setup the server
cargo run --release --bin zkcuda_setup

# prove a first instance
cargo run --release --bin zkcuda_prove
cargo run --release --bin zkcuda_verify

# prove a second instance
cargo run --release --bin zkcuda_prove
cargo run --release --bin zkcuda_verify

# shutdown the server
cargo run --release --bin zkcuda_cleanup
