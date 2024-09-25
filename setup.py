# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "tools",
        ["tools.pyx"],
        define_macros=[],  # No se necesitan macros adicionales
        extra_compile_args=["-O2"],  # Mantener optimización
        extra_link_args=[],  # Mantener vacío para evitar rutas duplicadas
    )
]

setup(
    name='tools',
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}  # Establecer nivel de lenguaje a Python 3
    ),
    zip_safe=False,
)
