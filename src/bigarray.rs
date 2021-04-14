use crate::conv::ToOCaml;
use crate::value::OCaml;
use crate::OCamlRuntime;
use core::marker::PhantomData;
use core::slice;
use ocaml_sys::bigarray;
use std::borrow::Borrow;

/// Bigarray kind
/// This is unsafe to implement, because it allows arbitrary casts
/// to the implementing type (through OCaml<Array1<T>>::as_slice()).
// Using a trait for this means a Rust type can only be matched to a
// single kind.  There are enough Rust types that this isn't a problem
// in practice.
pub unsafe trait BigarrayElt: Copy {
    /// OCaml bigarray type identifier
    const KIND: i32;
}

// TODO: when we have const-time panic and assert (RFC 1229),
// assert that size_of::<$t>() matches caml_ba_element_size[$k]
macro_rules! make_kind {
    ($t:ty, $k:ident) => {
        unsafe impl BigarrayElt for $t {
            const KIND: i32 = bigarray::Kind::$k as i32;
        }
    };
}

// In kind order
// Skips some kinds OCaml supports: caml_int, complex32, complex64
make_kind!(f32, FLOAT32);
make_kind!(f64, FLOAT64);
make_kind!(i8, SINT8);
make_kind!(u8, UINT8);
make_kind!(i16, SINT16);
make_kind!(u16, UINT16);
make_kind!(i32, INT32);
make_kind!(i64, INT64);
make_kind!(isize, NATIVE_INT);
make_kind!(char, CHAR);

// mlvalues.rs
pub struct Array1<A: BigarrayElt> {
    _marker: PhantomData<A>,
}

// memory.rs
/// Create a new OCaml `Bigarray.Array1` with the given type and size
///
/// Memory belongs to the OCaml GC,
/// including the data, which is in the malloc heap but will be freed on
/// collection through a custom block
pub fn alloc_bigarray1<'a, A: BigarrayElt>(
    cr: &'a mut OCamlRuntime,
    data: &[A],
) -> OCaml<'a, Array1<A>> {
    let len = data.len();
    let ocaml_ba;
    unsafe {
        // num_dims == 1
        // data == NULL, OCaml will allocate with malloc (outside the GC)
        // and add the CAML_BA_MANAGED flag
        // OCaml custom block contains a bigarray struct after the header,
        // that points to the data array
        ocaml_ba = bigarray::caml_ba_alloc_dims(A::KIND, 1, core::ptr::null_mut(), len);
        let ba_meta_ptr = ocaml_sys::field(ocaml_ba, 1) as *const bigarray::Bigarray;
        core::ptr::copy_nonoverlapping(data.as_ptr(), (*ba_meta_ptr).data as *mut A, len);
    }
    unsafe { OCaml::new(cr, ocaml_ba) }
}

// value.rs
impl<'a, A: BigarrayElt> OCaml<'a, Array1<A>> {
    /// Returns the number of items in `self`
    pub fn len(&self) -> usize {
        let ba = unsafe { self.custom_ptr_val::<bigarray::Bigarray>() };
        unsafe { *((*ba).dim.as_ptr() as *const usize) }
    }

    /// Returns true when `self.len() == 0`
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get underlying data as Rust slice
    pub fn as_slice(&self) -> &[A] {
        unsafe {
            let ba = self.custom_ptr_val::<bigarray::Bigarray>();
            slice::from_raw_parts((*ba).data as *const A, self.len())
        }
    }

    /// Get underlying data as mutable Rust slice
    ///
    /// # Safety
    ///
    /// OCaml values can alias, this breaks the borrow model
    // XXX just remove this
    unsafe fn _as_mut_slice(&mut self) -> &mut [A] {
        let ba = self.custom_ptr_val::<bigarray::Bigarray>();
        slice::from_raw_parts_mut((*ba).data as *mut A, self.len())
    }
}

// to_ocaml.rs
// This copies
unsafe impl<A: BigarrayElt> ToOCaml<Array1<A>> for &[A] {
    fn to_ocaml<'a>(&self, cr: &'a mut OCamlRuntime) -> OCaml<'a, Array1<A>> {
        alloc_bigarray1(cr, self)
    }
}

// Note: we deliberately don't implement FromOCaml<Array1<A>>,
// because this trait doesn't have a lifetime parameter
// and implementing would force a copy.
impl<'a, A: BigarrayElt> Borrow<[A]> for OCaml<'a, Array1<A>> {
    fn borrow(&self) -> &[A] {
        unsafe {
            let ba = self.custom_ptr_val::<bigarray::Bigarray>();
            slice::from_raw_parts((*ba).data as *const A, self.len())
        }
    }
}
