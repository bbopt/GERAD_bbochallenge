import os
import subprocess

# Importing Cython package
os.system("sudo apt-get install python3-pip")
os.system("pip install Cython")

# Setting environmental variable NOMAD_HOME with nomad.3.9.1 directory
path = os.getcwd()
path_nomad = path + '/nomad.3.9.1'
os.chdir(path_nomad)
os.environ['NOMAD_HOME'] = path_nomad
#os.environ['NOMAD_HOME'] = "/c/Users/Edward/Desktop/E20/GERAD_bbochallenge/nomad.3.9.1"  # Edward

# Run configure and make files
#subprocess.run(["bash.exe", "configure", "--compiler=/c/MinGW/bin/g++.exe"])  # Edward
subprocess.run(["bash.exe", "configure"])
os.system("g++  ./Makefile")  # TODO : essayer subprocess.call("g++ ./Makefile"), si fontionne pas

# Build PyNomad
os.chdir(path + '/interface')
os.environ['NOMAD_HOME_PERSONAL'] = path_nomad  # TODO : pas certain
os.system('python setup_PyNomad.py build_ext --inplace')

# Run test and return main directory
os.system('python runTest.py')
os.chdir(path)

