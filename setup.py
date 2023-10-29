from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='langdiff37',
      ext_modules=[cpp_extension.CppExtension('langdiff37', ['main.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})