from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='tscluster',
   version='1.0.1',
   description='A useful package for temporal clustering',
   license="MIT",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Jolomi Tosanwumi',
   author_email='tjolomi@gmail.com',
   #url="...",
   packages=find_packages(), 
   #python_requires='>=3.9.18',
   install_requires=[
       'numpy>=1.26', 
       'scipy>=1.10', 
       'gurobipy>=11.0', 
       'tslearn>=0.6.3',   
       'h5py>=3.10',
       'pandas>=2.2',
       'matplotlib>=3.8'
       ], #external packages as dependencies on TEST pypi
   #ToDo: on real pypi: ['numpy==1.26.4', 'scipy==1.12.0', 'gurobipy==11.0.1']. Also considering using >= a lowerbound
)