import numpy as np
import setuptools
from setuptools.extension import Extension

extensions = [
    Extension(
        'nn.clayers',
        ['nn/clayers.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'nn.clayers_v2',
        ['nn/clayers_v2.pyx'],
        include_dirs=[np.get_include()]
    )
]

setuptools.setup(
    name='numpy_neuron_network',
    version='0.1',
    description='numpy 构建神经网络',
    url='https://github.com/yizt/numpy_neuron_network',
    author='yizt',
    author_email='csuyzt@163.com',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'cython'],
    ext_modules=extensions,
    setup_requires=["cython>=0.28"]
)
