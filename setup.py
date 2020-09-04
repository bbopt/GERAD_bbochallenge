import os
import subprocess

# Setting environmental variable NOMAD_HOME with nomad.3.9.1 directory
path = os.getcwd()
path_nomad = path + '/nomad.3.9.1'
os.chdir(path_nomad)
os.environ['NOMAD_HOME'] = path_nomad  # TODO : fix pathing problem
#os.environ['NOMAD_HOME'] = path_nomad+'//bin'  # manual fix

# Run shell script configure and make
with open('configure', 'r') as file:
    configure_script = file.read()
    subprocess.run(configure_script, shell=True)

os.system("make")  # requires gcc compiler

# Build PyNomad
os.chdir(path + '/interface_block_of_8')
os.environ['NOMAD_HOME_PERSONAL'] = path_nomad
#os.system('python setup_PyNomad.py build_ext --inplace')
os.system('python setup_PyNomad.py install --user')
