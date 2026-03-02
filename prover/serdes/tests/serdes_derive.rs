use std::io::Cursor;

use serdes::ExpSerde;

#[derive(ExpSerde, Debug, PartialEq)]
struct TestStruct {
    x: u32,
    y: String,
}

#[derive(ExpSerde, Debug, PartialEq)]
enum TestEnum {
    Unit,
    Tuple(u32, String),
    Struct { x: u32, y: String },
}

#[test]
fn test_struct_serialization() {
    let original = TestStruct {
        x: 42,
        y: "hello".to_string(),
    };

    let mut buf = Vec::new();
    original.serialize_into(&mut buf).unwrap();

    let mut cursor = Cursor::new(buf);
    let deserialized = TestStruct::deserialize_from(&mut cursor).unwrap();

    assert_eq!(original, deserialized);
}

#[test]
fn test_enum_serialization() {
    let test_cases = vec![
        TestEnum::Unit,
        TestEnum::Tuple(42, "hello".to_string()),
        TestEnum::Struct {
            x: 42,
            y: "hello".to_string(),
        },
    ];

    for original in test_cases {
        let mut buf = Vec::new();
        original.serialize_into(&mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        let deserialized = TestEnum::deserialize_from(&mut cursor).unwrap();

        assert_eq!(original, deserialized);
    }
}
