#!/usr/bin/env python

"""The setup script."""

import sys
from setuptools import setup, find_packages, Extension
import re

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    requirements = [
        r
        for r in requirements
        if re.split("[^a-zA-Z0-9]", r)[0]
        not in [
            "python",
            "watchdog",
            "pip",
            "bump2version",
            "tox",
            "flake8",
            "imageio",
            "pytest",
            "wheel",
            "coverage",
            "twine",
        ]
    ]

test_requirements = ["pytest>=3"]

extensions = [
    Extension("pysamosa.model_helpers", ["pysamosa/model_helpers.pyx"]),
]

setup(
    author="Florian Schlembach",
    author_email="florian.schlembach@tum.de",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.11",
    ],
    description="This framework provides a Python implementation for retracking open ocean and coastal waveforms of SAR satellite altimetry, which are based on the open ocean power return echo waveform model SAMOSA2.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pysamosa",
    name="pysamosa",
    packages=find_packages(include=["pysamosa", "pysamosa.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    ext_modules=extensions,
    url="https://github.com/floschl/pysamosa",
    version="0.2.7",
    zip_safe=False,
    setup_requires=[],
)
