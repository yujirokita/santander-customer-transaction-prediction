import os
from time import sleep
import sys
argv = sys.argv

file = argv[1]
os.system(f'nohup python -u {file} > log/{file}.log &')

