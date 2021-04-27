#!/usr/bin/env python

from setuptools import setup, find_packages

# Get requirements
with open('requirements.txt') as f:
    requires = f.read().splitlines()

setup(name='abwoc',
      version='0.1',
      description='Pipeline for Astros but Without Cheating',
      author='Grosson, Meng, Tracy, White',
      packages=find_packages(),
      data_files=[('abwoc', ['abwoc/text.txt'])],
      install_requires=requires
     )
