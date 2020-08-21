import numpy
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="circuitlib",
    version="0.0.1",
    description="To easily simulate electrical circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=["matplotlib>=3.0", "scipy>=1.4.1", "numpy>=1.15",],
    ext_modules=cythonize(
        [
            Extension(
                "circuitlib.signal_generator_cython",
                ["circuitlib/signal_generator_cython.pyx"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                "circuitlib.differential",
                ["circuitlib/differential.pyx"],
                include_dirs=[numpy.get_include()],
            ),
        ]
    ),
    url="https://github.com/calamont/circuitlib",
    author="Callum Lamont",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
