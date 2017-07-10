#!/usr/bin/python
import sys

L1 = list()
L2 = list()
for line in open(sys.argv[1]):
	line = line.strip()
	line = line.strip("|")
	L1.append(line.split("|||"))
for line in open(sys.argv[2]):
	line = line.strip()
	line = line.strip("|")
	L2.append(line.split("|||"))

if len(L1) != len(L2):
	sys.stderr.write("len error\n")
	exit(1)

for i in range(len(L1)):
	if len(L1[i]) != len(L2[i]):
		sys.stderr.write(str(i)+" tok error\n")
		exit(1)
	for j in range(len(L1[i])):
		print L1[i][j].strip().replace("NULL","NULL-s"),
		print L2[i][j].strip().replace("NULL","NULL-e"),
		print "|||",
	print
