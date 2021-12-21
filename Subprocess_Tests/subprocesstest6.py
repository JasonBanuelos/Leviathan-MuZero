import subprocess

def run_command(command):
    '''
    Run a command on the shell and print the output.
    '''
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    for line in p.stdout.readlines():
        print(line),

    retval = p.wait()


# Open a file in vi.
command = 'vi subprocesstest4.py'
print(run_command(command))  # Print Vi's output to the screen.

# delete the first line of the file
command = 'dd'
print(run_command(command))

# save and quit out of the file
command = ':wq'
print(run_command(command))
