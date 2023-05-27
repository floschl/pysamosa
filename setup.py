#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Florian Schlembach",
    author_email='florian.schlembach@tum.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This framework provides a Python implementation for retracking open ocean and coastal waveforms of SAR satellite altimetry, which are based on the open ocean power return echo waveform model SAMOSA2.",
    entry_points={
        'console_scripts': [
            'pysamosa=pysamosa.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pysamosa',
    name='pysamosa',
    packages=find_packages(include=['pysamosa', 'pysamosa.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/floschl/pysamosa',
    version='0.1.0',
    zip_safe=False,
)
