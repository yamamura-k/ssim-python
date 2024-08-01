from setuptools import setup, Extension

from Cython.Build import cythonize
from numpy import get_include
from torch.utils import cpp_extension


setup(
    name="ssim_python",
    ext_modules=cythonize(
        [
            Extension(
                "lib_cython_0", sources=["ssim_python/lib/lib_cython_0.pyx"], include_dirs=[".", get_include()]
            ),
            Extension(
                "lib_cython_1", sources=["ssim_python/lib/lib_cython_1.pyx"], include_dirs=[".", get_include()]
            ),
            Extension(
                "lib_cython_2", sources=["ssim_python/lib/lib_cython_2.pyx"], include_dirs=[".", get_include()]
            ),
        ]
    ) + [cpp_extension.CppExtension("libcpp", ["ssim_python/lib/libcpp.cpp"], include_dirs=[cpp_extension.include_paths()])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

