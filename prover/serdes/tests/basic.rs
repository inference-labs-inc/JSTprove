use std::fmt::Debug;

use halo2curves::bn256::{G1Affine, G2Affine};
use serdes::ExpSerde;

fn test_serialize_deserialize_helper<T: ExpSerde + Debug + PartialEq>(obj: T) {
    let mut buf = Vec::new();
    obj.serialize_into(&mut buf).unwrap();
    let deserialized_obj = T::deserialize_from(&buf[..]).unwrap();
    assert_eq!(obj, deserialized_obj);
}

#[test]
#[ignore]
fn test_g1_g2_serialization() {
    let mut rng = rand::thread_rng();
    let g1 = G1Affine::random(&mut rng);
    let g2 = G2Affine::random(&mut rng);

    test_serialize_deserialize_helper(g1);
    test_serialize_deserialize_helper(g2);
}
