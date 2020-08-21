import os
import subprocess
from distutils.core import setup, Extension

# Setting environmental variable NOMAD_HOME with nomad.3.9.1 directory
path = os.getcwd()
path_nomad = path + '/nomad.3.9.1'
os.chdir(path_nomad)
#os.environ['NOMAD_HOME'] = path_nomad  # TODO : fix pathing problem
os.environ['NOMAD_HOME'] = path_nomad+'//bin'  # manual fix

# Run shell script configure and make
with open('configure', 'r') as file:
    configure_script = file.read()
subprocess.run(configure_script, shell=True)

os.system("make")

# Build PyNomad
#os.chdir(path + '/interface')
#os.environ['NOMAD_HOME_PERSONAL'] = path_nomad
#os.system('python setup_PyNomad.py build_ext --inplace')


