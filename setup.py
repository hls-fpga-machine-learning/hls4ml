from setuptools import setup
from setuptools import find_packages

import codecs
import os.path

def read(rel_path):
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(this_directory, rel_path), 'r') as fp:
        return fp.read()

setup(name='hls4ml',
      use_scm_version={"version_scheme": "no-guess-dev"},
      description='Machine learning in FPGAs using HLS',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      author='HLS4ML Team',
      author_email='hls4ml.help@gmail.com',
      url='https://github.com/hls-fpga-machine-learning/hls4ml',
      license='Apache 2.0',
      install_requires=[
          'numpy',
          'six',
          'pyyaml',
          'h5py',
          'onnx>=1.4.0',
          'calmjs.parse',
          'tabulate',
          'pydigitalwavetools==1.1',
          'qkeras',
      ],
      extras_require={
        'profiling': [
            'pandas',
            'seaborn',
            'matplotlib'
        ]
      },
      scripts=['scripts/hls4ml'],
      include_package_data=True,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: C++',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
