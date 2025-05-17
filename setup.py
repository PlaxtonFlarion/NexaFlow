#   ____       _
#  / ___|  ___| |_ _   _ _ __
#  \___ \ / _ \ __| | | | '_ \
#   ___) |  __/ |_| |_| | |_) |
#  |____/ \___|\__|\__,_| .__/
#                       |_|
#
# ==== Notes: License ====
# Copyright (c) 2024  Framix :: 画帧秀
# This file is licensed under the Framix :: 画帧秀 License. See the LICENSE.md file for more details.

from nexaflow import const
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name=const.NAME,
    version=const.VERSION,
    url=const.URL,
    author=const.AUTHOR,
    license=const.LICENSE,
    author_email=const.EMAIL,
    description=const.DESC,
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: JavaScript',
        'Programming Language :: Python :: 3.11',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows :: Windows 11',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: English',
    ]
)
