from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

pypi_classifiers = [
    'Programming Language :: Python :: 3',
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Operating System :: OS Independent",
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    "Topic :: Software Development :: Libraries :: Python Modules",
    'License :: OSI Approved :: MIT License',
]

install_requires = [
    'biopython>=1.70',
]

desc = """Predict ancestral sequence of fungal repeat elements by correcting for RIP-like mutations in multi-sequence DNA alignments."""

setup(name='derip2',
      version='0.0.1',
      description=desc,
      long_description=readme(),
      url='https://github.com/Adamtaranto/deRIP2',
      author='Adam Taranto',
      author_email='adam.taranto@anu.edu.au',
      license='MIT',
      packages=['derip2'],
      classifiers=pypi_classifiers,
      keywords=["Transposon","RIP","TE"],
      install_requires=install_requires,
      include_package_data=True,
      zip_safe=False,
      entry_points={
        'console_scripts': [
            'derip2=derip2.run_self:main',
        ],
    },
    )