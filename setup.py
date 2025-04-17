from setuptools import setup, find_packages

setup(
    name="pynsfem",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    author="MaximeRenault",
    author_email="renault.maxim@gmail.com",
    description="A Python library for Finite Element Method (FEM) computations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maximerenault/pynsfem",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
) 