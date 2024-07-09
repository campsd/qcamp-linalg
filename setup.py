#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
        name="qcamp_linalg",
        version="0.0.1",
        author="campsd",
        author_email="dcamps@lbl.gov",
        packages=["qcamp_linalg"],
        package_dir={"qcamp_linalg": "qcamp_linalg"},
        url="http://github.com/campsd/qcamp_linalg/",
        license="MIT",
        install_requires=["numpy >= 1.8",
                          "scipy >= 0.13",
                          "ipympl"]
)
