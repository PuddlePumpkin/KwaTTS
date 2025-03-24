import subprocess
import os
os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
print("\nStarting Piperbot process...\n")
subprocess.run(["python", "src/piperbot.py"])