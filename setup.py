import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL",
        "Operating System :: OS Independent",
    ),
    install_requires=['numpy', 'tensorflow', 'torch', 'toml', 'matplotlib', 'h5py']
)
