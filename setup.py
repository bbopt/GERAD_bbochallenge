import os
import subprocess
from distutils.core import setup, Extension


# Setting environmental variable NOMAD_HOME with nomad.3.9.1 directory
path = os.getcwd()
path_nomad = path + '/nomad.3.9.1'
os.chdir(path_nomad)
# os.environ['NOMAD_HOME'] = path_nomad
os.environ['NOMAD_HOME'] = "/c/Users/Edward/Desktop/E20/GERAD_bbochallenge/nomad.3.9.1"  # Edward


# Run shell script configure and make
subprocess.run(["bash.exe", "configure"])
# os.system("make")
os.system("C:\\MinGW\\bin\\mingw32-make.exe")  # Edward : simulates the "make" command for Windows


# Build PyNomad
os.chdir(path + '/interface')
os.environ['NOMAD_HOME_PERSONAL'] = path_nomad
os.system('python setup_PyNomad.py build_ext --inplace')


