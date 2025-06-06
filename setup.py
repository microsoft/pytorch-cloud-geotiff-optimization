# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Akram Zaytar",
    author_email='akramzaytar@microsoft.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Optimized PyTorch data loading pipeline for streaming Earth observation data from cloud storage to GPU, achieving 20× higher throughput and 85-95% GPU utilization through tile-aligned reads and optimized worker configurations.",
    entry_points={
        'console_scripts': [
            'ocogs=optimized_cog_streaming.ocogs:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='optimized_cog_streaming',
    name='optimized_cog_streaming',
    packages=find_packages(include=['optimized_cog_streaming', 'optimized_cog_streaming.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/akramz/optimized_cog_streaming',
    version='0.1.0',
    zip_safe=False,
)
