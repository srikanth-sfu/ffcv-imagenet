import os, time
import sys

while(True):
    os.system('zip -qq -r %s %s'%(sys.argv[1], sys.argv[2]))
    os.system('cp %s.zip /home/smuralid'%(sys.argv[1]))
    time.sleep(60)
