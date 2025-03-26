from setuptools import setup, find_packages

# # Read dependencies from requirements.txt
# required = []
# with open('requirements.txt') as f:
# 	required = f.read().splitlines()

setup(name='emforecaster',
	  version='0.0.1',
	  packages=find_packages(),
	  description='EMForecaster official code repository.',
      # requires=required,
	  author='Xavier Mootoo',
	  # license='MIT',install_requires=required,classifiers=['Programming Language :: Python :: 3','License :: OSI Approved :: MIT License','Operating System :: OS Independent',],
	  python_requires='>=3.10')
