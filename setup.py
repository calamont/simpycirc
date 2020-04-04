import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="simpycirc",
    version="0.0.1",
    description="To easily simulate electrical circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["docs", "tests"]),
    install_requires=[
        "matplotlib>=3.0",
        "scipy>=1.4.1",
        "numpy>=1.15",
        "pandas>=0.24.2",
    ],
    url="https://github.com/calamont/simpycirc",
    author="Callum Lamont",
    email="cal_lamont@hotmail.com",
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
