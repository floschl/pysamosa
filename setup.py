#!/usr/bin/env python

"""The setup script."""

import re
import sys

from setuptools import Extension, find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

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
            "black",
            "isort",
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="PySAMOSA is a software framework for processing open ocean and coastal waveforms from SAR satellite altimetry to measure sea surface heights, wave heights, and wind speed for the oceans and inland water bodies. Satellite altimetry is a space-borne remote sensing technique used for Earth observation.",
    install_requires=requirements,
    license="GNU Lesser General Public License v3 or later (LGPLv3+)",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="satellite altimetry retracking samosa+ samosa coastal open ocean sentinel esa"
    "sar altimetry ff-sar fully focused",
    name="pysamosa",
    packages=find_packages(include=["pysamosa", "pysamosa.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    ext_modules=extensions,
    url="https://github.com/floschl/pysamosa",
    version="0.5.11",
    zip_safe=False,
    setup_requires=[],
)
