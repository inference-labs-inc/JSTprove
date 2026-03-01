#[test]
fn test_mutually_exclusive_flags() {
    let mut enabled_ctr = 0;

    #[cfg(target_arch = "aarch64")]
    {
        enabled_ctr += 1;
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    {
        enabled_ctr += 1;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        enabled_ctr += 1;
    }

    assert_eq!(enabled_ctr, 1);
}
