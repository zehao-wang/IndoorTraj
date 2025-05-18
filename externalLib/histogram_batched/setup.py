from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")


setup(
    name="histogram_batched",
    ext_modules=[
        CUDAExtension(
            name="histogram_batched._C",
            sources=[
            "histogram_batched.cu",
            "ext.cpp"],
            extra_compile_args={"cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
