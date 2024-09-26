from setuptools import setup
from setuptools.extension import Extension
import numpy as np
from Cython.Distutils import build_ext

ext_modules=[
    Extension("ioniq.cparsers",["src/cparsers.pyx"],include_dirs=[np.get_include()] )
]

setup(
    name='ioniq',
    version='0.1',
    author='Ali Fallahi',
    author_email='fallahi.a@northeastern.edu',
    url='https://github.com/wanunulab/ioniq/',
    packages=['ioniq'],
    description='A modular nanopore data analysis package.',
    requires=[line.strip() for line in open('requirements.txt')],
    include=["src*"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
    
    
)
