from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('solve_parallel_mc.pyx')
)