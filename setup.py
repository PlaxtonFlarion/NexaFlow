from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nexaflow',
    version='0.1.0-beta',
    packages=find_packages(),
    url='https://github.com/PlaxtonFlarion/NexaFlow',
    license='MIT',
    author='AceKeppel',
    author_email='AceKeppel@outlook.com',
    description='NexaFlow',
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