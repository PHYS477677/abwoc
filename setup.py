#!/usr/bin/env python

from distutils.core import setup

# Get requirements
with open('requirements.txt') as f:
    requires = f.read().splitlines()

setup(name='abwoc',
      version='0.1',
      description='Pipeline for Astros but Without Cheating',
      author='Grosson, Meng, Tracy, White',
      packages=['abwoc'],
      install_requires=requires
     )