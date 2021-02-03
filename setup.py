# Copyright (C) 2021-2022 Ajay Arunachalam <ajay.arunachalam08@gmail.com>
# License: MIT, ajay.arunachalam08@gmail.com

from setuptools import setup
from msda.utils import version

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="msda",
    version=f"{version()}",
    description="MSDA - An open source, low-code time-series multi-sensor data analysis library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ajayarunachalam/msda",
    author="Ajay Arunachalam",
    author_email="ajay.arunachalam08@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["msda"],
    include_package_data=True,
    install_requires=required
)
