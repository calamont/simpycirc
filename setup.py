import setuptools

setuptools.setup(
    name="simpycirc",
    version="0.0.1",
    description="To easily simulate electrical circuits",
    packages=setuptools.find_packages(),
    install_requires=["matplotlib>=3.0", "scipy>=1.2", "pandas>=0.24.2"],
)
