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
        'matplotlib==3.8.1',
        'numpy==1.26.0',
        'pyiron_base==0.6.8',
        'scipy==1.11.3',
        'seaborn==0.13.0',
        'pyparsing==3.1.1',
    ],
    extras_require={
        'atomistic': [
            'ase==3.22.1',
            'pyiron_atomistics==0.3.5',
            'pycp2k==0.2.2',
        ],
        'fenics': [
            'fenics==2019.1.0',
            'mshr==2019.1.0',
        ],
        'image': ['scikit-image==0.22.0'],
        'generic': [
            'boto3==1.28.78', 
            'moto==4.2.7'
        ],
        'tinybase': [
            'distributed==2023.10.1',
            'pympipool==0.7.3'
        ]
    },
    cmdclass=versioneer.get_cmdclass(),
    
)
