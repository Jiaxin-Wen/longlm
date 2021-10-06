import os
import subprocess

def verify():
    with open(os.devnull, 'wb') as devnull:
        try:
            subprocess.check_call('ninja --version'.split(), stdout=devnull)
        except OSError:
            return False
        else:
            return True

print(verify())