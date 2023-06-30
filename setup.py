from setuptools import find_packages, setup

setup(
    name="custom_module",
    packages=find_packages(include=["custom_module"]),
)
