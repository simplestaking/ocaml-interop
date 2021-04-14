// Copyright (c) SimpleStaking and Tezedge Contributors
// SPDX-License-Identifier: MIT

extern crate ocaml_interop;

use ocaml_interop::{OCaml, OCamlBytes, OCamlRuntime, ToOCaml};

#[cfg(test)]
use ocaml_interop::{bigarray, BoxRoot};

mod ocaml {
    use ocaml_interop::*;

    pub struct TestRecord {
        pub i: i64,
        pub f: f64,
        pub i32: i32,
        pub i64: Box<i64>,
        pub s: String,
        pub t: (i64, f64),
    }

    pub enum Movement {
        Step(i64),
        RotateLeft,
        RotateRight,
    }

    pub enum PolymorphicEnum {
        Unit,
        Single(f64),
        Multiple(i64, String),
    }

    impl_to_ocaml_record! {
        TestRecord {
            i: OCamlInt,
            f: OCamlFloat,
            i32: OCamlInt32,
            i64: OCamlInt64,
            s: String,
            t: (OCamlInt, OCamlFloat),
        }
    }

    impl_to_ocaml_variant! {
        Movement {
            Movement::Step(count: OCamlInt),
            Movement::RotateLeft,
            Movement::RotateRight,
        }
    }

    impl_to_ocaml_polymorphic_variant! {
        PolymorphicEnum {
            PolymorphicEnum::Single(i: OCamlFloat),
            PolymorphicEnum::Multiple(i: OCamlInt, s: String),
            PolymorphicEnum::Unit,
        }
    }

    ocaml! {
        pub fn increment_bytes(bytes: String, first_n: OCamlInt) -> String;
        pub fn increment_ints_list(ints: OCamlList<OCamlInt>) -> OCamlList<OCamlInt>;
        pub fn twice(num: OCamlInt) -> OCamlInt;
        pub fn make_tuple(fst: String, snd: OCamlInt) -> (String, OCamlInt);
        pub fn make_some(value: String) -> Option<String>;
        pub fn make_ok(value: OCamlInt) -> Result<OCamlInt, String>;
        pub fn make_error(value: String) -> Result<OCamlInt, String>;
        pub fn stringify_record(record: TestRecord) -> String;
        pub fn stringify_variant(variant: Movement) -> String;
        pub fn stringify_polymorphic_variant(pvariant: PolymorphicEnum) -> String;
        pub fn raises_message_exception(message: String);
        pub fn raises_nonmessage_exception(unit: ());
        pub fn raises_nonblock_exception(unit: ());
        pub fn double_u16_array(array: bigarray::Array1<u16>);
    }
}

pub fn increment_bytes(cr: &mut OCamlRuntime, bytes: &str, first_n: usize) -> String {
    let bytes = bytes.to_boxroot(cr);
    let first_n = unsafe { OCaml::of_i64_unchecked(first_n as i64) };
    let result = ocaml::increment_bytes(cr, &bytes, &first_n);
    result.to_rust(cr)
}

pub fn increment_ints_list(cr: &mut OCamlRuntime, ints: &Vec<i64>) -> Vec<i64> {
    let ints = ints.to_boxroot(cr);
    let result = ocaml::increment_ints_list(cr, &ints);
    result.to_rust(cr)
}

pub fn twice(cr: &mut OCamlRuntime, num: i64) -> i64 {
    let num = unsafe { OCaml::of_i64_unchecked(num) };
    let result = ocaml::twice(cr, &num);
    result.to_rust(cr)
}

pub fn make_tuple(cr: &mut OCamlRuntime, fst: String, snd: i64) -> (String, i64) {
    let num = unsafe { OCaml::of_i64_unchecked(snd) };
    let str = fst.to_boxroot(cr);
    let result = ocaml::make_tuple(cr, &str, &num);
    result.to_rust(cr)
}

pub fn make_some(cr: &mut OCamlRuntime, value: String) -> Option<String> {
    let str = value.to_boxroot(cr);
    let result = ocaml::make_some(cr, &str);
    result.to_rust(cr)
}

pub fn make_ok(cr: &mut OCamlRuntime, value: i64) -> Result<i64, String> {
    let value = unsafe { OCaml::of_i64_unchecked(value) };
    let result = ocaml::make_ok(cr, &value);
    result.to_rust(cr)
}

pub fn make_error(cr: &mut OCamlRuntime, value: String) -> Result<i64, String> {
    let result = value.to_boxroot(cr);
    let result = ocaml::make_error(cr, &result);
    result.to_rust(cr)
}

pub fn verify_record_test(cr: &mut OCamlRuntime, record: ocaml::TestRecord) -> String {
    let ocaml_record = record.to_boxroot(cr);
    let result = ocaml::stringify_record(cr, &ocaml_record);
    result.to_rust(cr)
}

pub fn verify_variant_test(cr: &mut OCamlRuntime, variant: ocaml::Movement) -> String {
    let ocaml_variant = variant.to_boxroot(cr);
    let result = ocaml::stringify_variant(cr, &ocaml_variant);
    result.to_rust(cr)
}

pub fn verify_polymorphic_variant_test(cr: &mut OCamlRuntime, variant: ocaml::PolymorphicEnum) -> String {
    let ocaml_variant = variant.to_boxroot(cr);
    let result = ocaml::stringify_polymorphic_variant(cr, &ocaml_variant);
    result.to_rust(cr)
}

pub fn allocate_alot(cr: &mut OCamlRuntime) -> bool {
    let vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    for _n in 1..50000 {
        let _x: OCaml<OCamlBytes> = vec.to_ocaml(cr);
        let _y: OCaml<OCamlBytes> = vec.to_ocaml(cr);
        let _z: OCaml<OCamlBytes> = vec.to_ocaml(cr);
    }
    true
}

// Tests

// NOTE: required because at the moment, no synchronization is done on OCaml calls
#[cfg(test)]
use serial_test::serial;

#[test]
#[serial]
fn test_twice() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(twice(&mut cr, 10), 20);
}

#[test]
#[serial]
fn test_increment_bytes() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(
        increment_bytes(&mut cr, "0000000000000000", 10),
        "1111111111000000"
    );
}

#[test]
#[serial]
fn test_increment_ints_list() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    let ints = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let expected = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(increment_ints_list(&mut cr, &ints), expected);
}

#[test]
#[serial]
fn test_make_tuple() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(
        make_tuple(&mut cr, "fst".to_owned(), 9),
        ("fst".to_owned(), 9)
    );
}

#[test]
#[serial]
fn test_make_some() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(
        make_some(&mut cr, "some".to_owned()),
        Some("some".to_owned())
    );
}

#[test]
#[serial]
fn test_make_result() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(make_ok(&mut cr, 10), Ok(10));
    assert_eq!(
        make_error(&mut cr, "error".to_owned()),
        Err("error".to_owned())
    );
}

#[test]
#[serial]
fn test_frame_management() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(allocate_alot(&mut cr), true);
}

#[test]
#[serial]
fn test_record_conversion() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    let record = ocaml::TestRecord {
        i: 10,
        f: 5.0,
        i32: 10,
        i64: Box::new(10),
        s: "string".to_owned(),
        t: (10, 5.0),
    };
    let expected = "{ i=10; f=5.00; i32=10; i64=10; s=string; t=(10, 5.00) }".to_owned();
    assert_eq!(verify_record_test(&mut cr, record), expected);
}

#[test]
#[serial]
fn test_variant_conversion() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(
        verify_variant_test(&mut cr, ocaml::Movement::RotateLeft),
        "RotateLeft".to_owned()
    );
    assert_eq!(
        verify_variant_test(&mut cr, ocaml::Movement::RotateRight),
        "RotateRight".to_owned()
    );
    assert_eq!(
        verify_variant_test(&mut cr, ocaml::Movement::Step(10)),
        "Step(10)".to_owned()
    );
}

#[test]
#[serial]
fn test_polymorphic_variant_conversion() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };
    assert_eq!(
        verify_polymorphic_variant_test(&mut cr, ocaml::PolymorphicEnum::Unit),
        "Unit".to_owned()
    );
    assert_eq!(
        verify_polymorphic_variant_test(&mut cr, ocaml::PolymorphicEnum::Single(10.0)),
        "Single(10.00)".to_owned()
    );
    assert_eq!(
        verify_polymorphic_variant_test(&mut cr, ocaml::PolymorphicEnum::Multiple(10, "text".to_string())),
        "Multiple(10, text)".to_owned()
    );
}

#[test]
#[serial]
fn test_bigarray() {
    OCamlRuntime::init_persistent();
    let mut cr = unsafe { OCamlRuntime::recover_handle() };

    let arr: Vec<u16> = (0..16).collect();

    let crr = &mut cr;
    let arr_ocaml: BoxRoot<bigarray::Array1<_>> = arr.as_slice().to_boxroot(crr);
    ocaml::double_u16_array(crr, &arr_ocaml);
    assert_eq!(
        crr.get(&arr_ocaml).as_slice(),
        (0..16u16).map(|i| i * 2).collect::<Vec<_>>().as_slice()
    );
}

#[test]
#[serial]
fn test_exception_handling_with_message() {
    OCamlRuntime::init_persistent();
    let result = std::panic::catch_unwind(move || {
        let mut cr = unsafe { OCamlRuntime::recover_handle() };
        let mcr = &mut cr;
        let message = "my-error-message".to_boxroot(mcr);
        ocaml::raises_message_exception(mcr, &message);
    });
    assert_eq!(
        result
            .err()
            .and_then(|err| Some(err.downcast_ref::<String>().unwrap().clone()))
            .unwrap(),
        "OCaml exception, message: Some(\"my-error-message\")"
    );
}

#[test]
#[serial]
fn test_exception_handling_without_message() {
    OCamlRuntime::init_persistent();
    let result = std::panic::catch_unwind(|| {
        let cr = unsafe { OCamlRuntime::recover_handle() };
        ocaml::raises_nonmessage_exception(cr, &OCaml::unit());
    });
    assert_eq!(
        result
            .err()
            .and_then(|err| Some(err.downcast_ref::<String>().unwrap().clone()))
            .unwrap(),
        "OCaml exception, message: None"
    );
}

#[test]
#[serial]
fn test_exception_handling_nonblock_exception() {
    OCamlRuntime::init_persistent();
    let result = std::panic::catch_unwind(|| {
        let cr = unsafe { OCamlRuntime::recover_handle() };
        ocaml::raises_nonblock_exception(cr, &OCaml::unit());
    });
    assert_eq!(
        result
            .err()
            .and_then(|err| Some(err.downcast_ref::<String>().unwrap().clone()))
            .unwrap(),
        "OCaml exception, message: None"
    );
}
