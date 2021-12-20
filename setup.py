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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[        
        'matplotlib==3.5.1',
        'numpy==1.21.4',
        'pyiron_base==0.4.3',
        'scipy==1.7.3',
        'seaborn==0.11.2',
        'pyparsing==3.0.6'
    ],
    extras_require={
        'atomistic': [
            'ase==3.22.1',
            'pyiron_atomistics==0.2.33',
        ],
        'fenics': [
            'fenics==2019.1.0',
            'mshr==2019.1.0',
        ],
        'image': ['scikit-image==0.19.1'],
        'generic': [
            'boto3==1.20.24', 
            'moto==2.2.19'
        ],
    },
    cmdclass=versioneer.get_cmdclass(),
    
)
