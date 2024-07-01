from setuptools import find_packages, setup

setup(name='QAS_parameterizedGate_gym',
      version='0.0.1',
      install_requires=['gym', 'qulacs', 'numpy'],
      packages=find_packages('src'),
      package_dir={'': 'src'})
