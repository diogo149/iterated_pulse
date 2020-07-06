from setuptools import setup

setup(
    name='dnnlib',

    version='0.0.1',

    description='dnnlib - from StyleGAN repository',

    url='https://github.com/podgorskiy/dnnlib',

    author='NVIDIA',

    license='Creative Commons Attribution-NonCommercial 4.0',

    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=['dnnlib', 'dnnlib.submission', 'dnnlib.submission._internal', 'dnnlib.tflib'],
)
