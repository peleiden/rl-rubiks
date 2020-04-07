// https://bheisler.github.io/post/calling-rust-in-python/

#[no_mangle]
pub extern "C" fn doub(x: i32) -> i32 {
    x * 2
}

#[no_mangle]
pub extern "C" fn sum(x: Vec<i32>) -> i32 {
    println!("{:?}", x);
    let mut s = 0;
    for elem in x {
        s += elem;
    }
    s + 1
}


