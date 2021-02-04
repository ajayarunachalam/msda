# Copyright (C) 2021-2022 Ajay Arunachalam <ajay.arunachalam08@gmail.com>
# License: MIT, ajay.arunachalam08@gmail.com

from distutils.core import setup
import setuptools

__version__ = '1.0.3'

def readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


setuptools.setup(
    name='msda',
    version=__version__,
    description='MSDA - An open source, low-code time-series multi-sensor data analysis library in Python.',
    long_description = readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/ajayarunachalam/msda',
    install_requires=['matplotlib', 'numpy', 'datetime', 'pandas', 'statistics'],
    license='MIT',
    include_package_data=True,
    author='Ajay Arunachalam',
    author_email='ajay.arunachalam08@gmail.com')
