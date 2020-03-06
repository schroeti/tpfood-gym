from setuptools import setup, find_packages

setup(name='gym_tpfood',
      version='0.0.1',
      install_requires=['gym', 
                       'numpy'],
       packages=find_packages(), # And any other dependencies foo needs
     ) 
