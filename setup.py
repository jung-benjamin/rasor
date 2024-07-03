#! /usr/bin/env python3

from setuptools import find_packages, setup

setup(name='rasor',
      version='0.2.0',
      author='Benjamin Jung',
      license='BSD-3-Clause',
      packages=find_packages(include=['rasor', 'rasor.*']),
      description='Algorithms for selecting suitable isotopic ratios.',
      entry_points={
          'console_scripts': [
              'run_ratio_selection=rasor.command_line:run_ratio_selection',
              'run_pre_selection=rasor.command_line:run_pre_selection',
              'run_post_processing=rasor.command_line:run_post_processing'
          ]
      })
