import os
os.system("ls -l")
os.system("nvidia-smi")
# os.system("sudo su")
# os.system("sudo apt-get update")


# import subprocess
# test = subprocess.Popen(["ping","-W","2","-c", "1", "192.168.1.70"], stdout=subprocess.PIPE)
# output = test.communicate()[0]
#
# from subprocess import call
# # call(["ls", "-l"])
# call(["sudo", "su"])


import subprocess
subprocess.check_output(["sudo", "apt-get update", "params"])