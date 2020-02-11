import os, sys
os.chdir(sys.path[0])
from cffi import FFI
import numpy as np

ffi = FFI()
ffi.cdef("""
	int doub(int);
	int sum(int *);
""")

C = ffi.dlopen("pyrs/target/debug/libpyrs.so")
print(C.doub(5))

a = np.ones(40).astype(int) * 0
print(sys.getsizeof(a))
dat = a.__array_interface__["data"][0]
cptr = ffi.cast("int*", dat)
print(cptr)
print(C.sum(cptr))
