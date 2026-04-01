use crate::bn256::{Fr, FR_TABLE};

#[test]
fn test_table() {
    for (i, e) in FR_TABLE.iter().enumerate() {
        assert_eq!(Fr::from(i as u64), *e);
    }
}
