#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="Tensorflow Archive",
      version="0.1.0",
      description="Tensorflow Archive",
      author="Gokhan Kesler",
      packages=find_packages(include=["src", 'src.*']),
      author_email="gokhankesler@gmail.com",
      install_requires=["matplotlib",
                        "numpy",
                        "scipy",
                        "tensorflow"
                        ],
      )
