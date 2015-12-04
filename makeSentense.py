#!/usr/bin/env python

import sys

fin = open(sys.argv[1])

fout = open(sys.argv[2],"w")

l = 0
while True:
	line = fin.readline()
	if line in ("", None):
		break
	if len(line.split()) < 3:
		continue
	if not line.split()[2] in ("@card@","<unknown>"):
		fout.write(line.split()[2]+" ")
	l += 1
	if l == 100:
		fout.write("\n")
	
fout.close()
