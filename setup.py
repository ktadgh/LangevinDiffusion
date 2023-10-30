from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='langdiff65',
    ext_modules=[CppExtension(
        name='langdiff65',
        sources=['main.cpp'],
        extra_compile_args=['-Ofast','-funroll-loops','-fopt-info-vec-missed'] # Compiler options, adjust as needed
        #extra_cflags=['-std=c++17']  # Additional compiler flags
    )]
    ,cmdclass={'build_ext': BuildExtension}
)
    
    