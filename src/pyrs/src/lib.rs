// https://bheisler.github.io/post/calling-rust-in-python/

#[no_mangle]
pub extern "C" fn doub(x: i32) -> i32 {
    x * 2
}


