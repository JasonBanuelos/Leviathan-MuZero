"""Create a function that accepts a string as an input, enters the string into the terminal, and returns the output of the terminal as a string. This function should be able to open command line programs and editors such as Vi, and return the output of those programs as a string."""

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
