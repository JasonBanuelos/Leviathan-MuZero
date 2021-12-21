import subprocess
import re

while True:
    # prevents lots of python error output
    try:
        s = raw_input('> ')
    except:
        break

    # check if you should exit
    if s.strip().lower() == 'exit':
        break

    # try to run command
    try:
        cmd = subprocess.Popen(re.split(r'\s+', s), stdout=subprocess.PIPE)
        cmd_out = cmd.stdout.read()

        # Process output
        print(cmd_out)

    except OSError:
        print('Invalid command')

