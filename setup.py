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
        'Programming Language :: Python :: 3.12',
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[        
        'matplotlib==3.9.2',
        'numpy==1.26.4',
        'pyiron_snippets==0.1.3',
        'pyiron_base==0.10.1',
        'scipy==1.14.1',
        'seaborn==0.13.2',
        'pyparsing==3.1.4',
    ],
    extras_require={
        'atomistic': [
            'ase==3.23.0',
            'pyiron_atomistics==0.6.12',
            'pycp2k==0.2.2',
        ],
        'fenics': [
            'fenics==2019.1.0',
            'mshr==2019.1.0',
        ],
        'image': ['scikit-image==0.24.0'],
        'generic': [
            'boto3==1.35.86', 
            'moto==5.0.16'
        ],
        'tinybase': [
            'distributed==2024.8.0',
            'pymatgen==2024.9.17.1',
            'pympipool==0.8.4',
            'h5io_browser==0.1.1',
        ]
    },
    cmdclass=versioneer.get_cmdclass(),
)
