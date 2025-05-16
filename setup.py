from setuptools import setup, find_packages

setup(
    name="ledalabpy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.9",
    author="Vignesh Ravichandran",
    author_email="vignesh_ravi@uri.edu",
    description="Python port of Ledalab MATLAB software for EDA analysis",
    keywords="eda, gsr, electrodermal, psychophysiology",
    url="https://github.com/viggi1000/ledalabpy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
