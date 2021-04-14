(* Copyright (c) SimpleStaking and Tezedge Contributors
   SPDX-License-Identifier: MIT *)

type test_record = {
  i: int;
  f: float;
  i32: int32;
  i64: int64;
  s: string;
  t: int * float;
}

type movement =
  | Step of int
  | RotateLeft
  | RotateRight

type polymorphic_enum = [
  | `Unit
  | `Single of float
  | `Multiple of (int * string)
]

let increment_bytes bytes first_n =
  let limit = (min (Bytes.length bytes) first_n) - 1 in
  for i = 0 to limit do
    let value = (Bytes.get_uint8 bytes i) + 1 in
    Bytes.set_uint8 bytes i value
  done;
  bytes

let decrement_bytes bytes first_n =
  let limit = (min (Bytes.length bytes) first_n) - 1 in
  for i = 0 to limit do
    let value = (Bytes.get_uint8 bytes i) - 1 in
    Bytes.set_uint8 bytes i value
  done;
  bytes

let increment_ints_list ints =
  List.map ((+) 1) ints

let twice x = 2 * x

let make_tuple a b = (a, b)

let make_some x = Some x

let make_ok x = Ok x

let make_error x = Error x

let stringify_record { i; f; i32; i64; s; t = (t1, t2); } =
  Printf.sprintf "{ i=%d; f=%.2f; i32=%ld; i64=%Ld; s=%s; t=(%d, %.2f) }"
    i f i32 i64 s t1 t2

let stringify_variant = function
  | RotateLeft -> "RotateLeft"
  | RotateRight -> "RotateRight"
  | Step n -> Printf.sprintf "Step(%d)" n

let stringify_polymorphic_variant = function
  | `Single n -> Printf.sprintf "Single(%.2f)" n
  | `Multiple (n, s) -> Printf.sprintf "Multiple(%d, %s)" n s
  | `Unit -> "Unit"

let raises_message_exception msg = failwith msg

let raises_nonblock_exception () = raise Not_found

exception WithInt of int

let raises_nonmessage_exception () = raise (WithInt 10)

let double_u16_array arr =
  let open Bigarray in
  let n = Array1.dim arr in
  for i = 0 to pred n do
    Array1.(set arr i ((get arr i) * 2))
  done


let () =
  Callback.register "increment_bytes" increment_bytes;
  Callback.register "decrement_bytes" decrement_bytes;
  Callback.register "increment_ints_list" increment_ints_list;
  Callback.register "twice" twice;
  Callback.register "make_tuple" make_tuple;
  Callback.register "make_some" make_some;
  Callback.register "make_ok" make_ok;
  Callback.register "make_error" make_error;
  Callback.register "stringify_record" stringify_record;
  Callback.register "stringify_variant" stringify_variant;
  Callback.register "stringify_polymorphic_variant" stringify_polymorphic_variant;
  Callback.register "raises_message_exception" raises_message_exception;
  Callback.register "raises_nonmessage_exception" raises_nonmessage_exception;
  Callback.register "raises_nonblock_exception" raises_nonblock_exception;
  Callback.register "double_u16_array" double_u16_array;
