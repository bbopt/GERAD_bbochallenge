import os
import subprocess
from distutils.core import setup, Extension

# Setting environmental variable NOMAD_HOME with nomad.3.9.1 directory
path = os.getcwd()
path_nomad = path + '/nomad.3.9.1'
os.environ['NOMAD_HOME'] = path_nomad
os.chdir(path_nomad)

# Run shell script configure and make
subprocess.run("configure", shell=True)
os.system("make Makefile")

# Build PyNomad
os.chdir(path + '/interface')
os.environ['NOMAD_HOME_PERSONAL'] = path_nomad
os.system('python setup_PyNomad.py build_ext --inplace')


