import os
from time import sleep

import sys
argv = sys.argv

import datetime as dt
now = dt.datetime.now().strftime('%Y%m%d%H%M%S')

file = argv[1]
os.system(f'nohup python -u {file} > log/{file}.{now}.log &')

