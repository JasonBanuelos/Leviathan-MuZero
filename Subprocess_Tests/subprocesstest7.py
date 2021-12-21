import pexpect

def terminal(input):
	child = pexpect.spawn('/bin/bash')
	child.sendline(input)
	terminal_state = child.read()
	return terminal_state

terminal("vi subprocesstest4.py")
terminal(":q!")
