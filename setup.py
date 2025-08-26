from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ionique.cparsers", ["src/ionique/cparsers.pyx"], include_dirs=[np.get_include()])
]

setup(
    ext_modules=cythonize(extensions)
)
