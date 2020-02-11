from cffi import FFI

ffi = FFI()
ffi.cdef("""
    int doub(int);
""")

C = ffi.dlopen("pyrs/target/debug/libpyrs.so")

print(C.doub(9))
