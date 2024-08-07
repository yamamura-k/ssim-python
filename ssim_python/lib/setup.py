from setuptools import setup as setup_t
from distutils.core import Extension, setup

from Cython.Build import cythonize
from numpy import get_include
from torch.utils import cpp_extension

ext = Extension(
    "lib_cython_0", sources=["lib_cython_0.pyx"], include_dirs=[".", get_include()]
)
setup(name="lib_cython_0", ext_modules=cythonize([ext]))

ext = Extension(
    "lib_cython_1", sources=["lib_cython_1.pyx"], include_dirs=[".", get_include()]
)
setup(name="lib_cython_1", ext_modules=cythonize([ext]))

ext = Extension(
    "lib_cython_2", sources=["lib_cython_2.pyx"], include_dirs=[".", get_include()]
)
setup(name="lib_cython_2", ext_modules=cythonize([ext]))

setup_t(
    name="libcpp",
    ext_modules=[cpp_extension.CppExtension("libcpp", ["libcpp.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

