import setuptools
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='MLkit',
    version='0.0.3dev',
    author='CWhy',
    author_email='chenyu.nus@gmail.com',
    description='Some home-brew tools for machine learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/cwhy/MLkit',
    packages=setuptools.find_packages(),
    classifiers=(
        "Development Status :: 3 - Alpha",
        "Intended Audience :: CWhy and his friends",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GPL",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.6',
    install_requires=['numpy', 'tensorflow-gpu', 'torch', 'toml', 'matplotlib', 'h5py']
)
