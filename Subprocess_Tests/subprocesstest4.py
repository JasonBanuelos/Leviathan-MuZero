import subprocess

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    return process.communicate()[0]
    
"""Use this function to open a file called 'subprocesstest4.py' in Vi, delete the first line in that file using Vu, and then save the file and close Vi."""

command = 'vi subprocesstest4.py'
run_command(command)

# delete the first line of the file
command = 'dd'
run_command(command)

# save and quit out of the file
command = ':wq'
run_command(command)
