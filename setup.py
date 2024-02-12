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
        'matplotlib==3.8.2',
        'numpy==1.26.3',
        'pyiron_base==0.7.4',
        'scipy==1.12.0',
        'seaborn==0.13.2',
        'pyparsing==3.1.1',
    ],
    extras_require={
        'atomistic': [
            'ase==3.22.1',
            'pyiron_atomistics==0.4.13',
            'pycp2k==0.2.2',
        ],
        'fenics': [
            'fenics==2019.1.0',
            'mshr==2019.1.0',
        ],
        'image': ['scikit-image==0.22.0'],
        'generic': [
            'boto3==1.34.34', 
            'moto==5.0.1'
        ],
        'tinybase': [
            'distributed==2024.2.0',
            'pympipool==0.7.11',
            'h5io_browser==0.0.7',
        ]
    },
    cmdclass=versioneer.get_cmdclass(),
)
