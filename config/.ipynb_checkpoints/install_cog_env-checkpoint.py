import subprocess

subprocess.run(["sudo", "curl", "-o", "/usr/local/bin/cog", "-L", "https://github.com/replicate/cog/releases/latest/download/cog_uname -s_uname -m"])
subprocess.run(["sudo", "chmod", "+x", "/usr/local/bin/cog"])
subprocess.run(["cog", "init"])