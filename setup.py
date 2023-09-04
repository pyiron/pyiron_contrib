"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer


setup(
    name='pyiron_contrib',
    version=versioneer.get_version(),
    description='Repository for user-generated plugins to the pyiron IDE.',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/pyiron_contrib',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='huber@mpie.de',
    license='BSD',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[        
        'matplotlib==3.7.2',
        'numpy==1.24.3',
        'pyiron_base==0.6.3',
        'scipy==1.11.1',
        'seaborn==0.12.2',
        'pyparsing==3.0.9',
    ],
    extras_require={
        'atomistic': [
            'ase==3.22.1',
            'pyiron_atomistics==0.3.0',
            'pycp2k==0.2.2',
        ],
        'executors': [
            'cloudpickle',
        ],
        'fenics': [
            'fenics==2019.1.0',
            'mshr==2019.1.0',
        ],
        'image': ['scikit-image==0.21.0'],
        'generic': [
            'boto3==1.28.25', 
            'moto==4.1.14'
        ],
        'workflow': [
            'cloudpickle',
            'python>=3.10',
            'graphviz',
            'typeguard==4.1.0'
        ],
        'tinybase': [
            'distributed==2023.9.0',
            'pympipool==0.6.2'
        ]
    },
    cmdclass=versioneer.get_cmdclass(),
    
)
