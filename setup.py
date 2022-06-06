from setuptools import setup
from setuptools.extension import Extension
import numpy as np
from Cython.Distutils import build_ext

ext_modules=[
    Extension("PythIon.Parsers.cparsers",["PythIon/Parsers/cparsers.pyx"],include_dirs=[np.get_include()] )
]

setup(
    name='PythIon',
    version='0.3.0',
    author='Ali Fallahi',
    author_email='fallahi.a@northeastern.edu',
    packages=['PythIon'],
    description='Nanopore Data Analysis package.',
    requires=['cython'],
    include=["PythIon*"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
    
    
)