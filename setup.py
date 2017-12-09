from distutils.core import setup

setup(
    name='MLkit',
    version='0.0.1dev',
    packages=['MLkit'],
    url='',
    license='GPL',
    author='CWhy',
    author_email='chenyu.nus@gmail.com',
    description='Some home-brew tools for machine learning',
    requires=['numpy', 'tensorflow', 'torch']
)
