from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="libcpp",
    ext_modules=[cpp_extension.CppExtension("libcpp", ["libcpp.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
