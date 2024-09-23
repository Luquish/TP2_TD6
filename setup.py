# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='custom_tools',
    ext_modules=cythonize("tools.pyx"),
    zip_safe=False,
)
