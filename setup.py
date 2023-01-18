from setuptools import setup, find_packages

setup(
    name='nuccorrelation',
    version='1.0',
    packages=find_packages(include=["nuccorrelation"]),
    test_suite='tests_package',
    install_requires=[
        "matplotlib>=3.6.3",
        "matplotlib-inline>=0.1.6",
        "numpy<1.23.0,>=1.16.5",
        "openpyxl>=3.0.10",
        "pandas>=1.5.2",
        "seaborn>=0.12.1"
    ]
)
