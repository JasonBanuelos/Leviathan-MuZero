import subprocess

output = subprocess.check_output("ls", shell=True)
#output += subprocess.check_output("vi ostest.py", shell=True)  # Same as subprocess.run("vi ostest.py", shell=True, stdout=subprocess.PIPE).stdout
# There's got to be a less convoluted way to break this thing into multiple lines than this.
split_output = output.split(b"\n")
string_output = ""
for s in split_output:
	tempstring = str(s)
	tempstring = tempstring[:-1]
	tempstring = tempstring[2:]
	string_output += tempstring + "\n"
string_output = string_output[:-2]
print(string_output)
