import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

# Custom build extension subclass to detect compiler type at runtime
class CustomBuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        
        # Apply OpenMP flags based on compile backend
        if compiler_type == "msvc":
            omp_compile_args = ["/openmp"]
            omp_link_args = []
        else:
            # unix, mingw32, clang, etc.
            omp_compile_args = ["-fopenmp"]
            omp_link_args = ["-fopenmp"]
            
        for ext in self.extensions:
            ext.extra_compile_args.extend(omp_compile_args)
            ext.extra_link_args.extend(omp_link_args)
            
        super().build_extensions()

# Define Cython compilation extensions (without hardcoding OS compile arguments)
extensions = [
    Extension(
        name="ai_engine.features.openmp_backend",
        sources=["ai_engine/features/openmp/openmp_backend.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[],
        extra_link_args=[],
    )
]

setup(
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True
        }
    )
)
