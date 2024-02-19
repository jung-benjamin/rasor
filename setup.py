#! /usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='rasor',
    version='0.1.0',
    author='Benjamin Jung',
    license='BSD-3-Clause',
    packages=find_packages(include=['rasor', 'rasor.*']),
    description='Algorithms for selecting suitable isotopic ratios.',
)
