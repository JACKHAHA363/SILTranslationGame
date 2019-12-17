import sys
import time
import os
import re
import subprocess
import pprint

for idx in range(97):
    cmd = "scancel --name 180603_{}".format(idx)
    subprocess.call(cmd, shell=True)
    time.sleep(0.001)

