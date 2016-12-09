#!/usr/bin/python

import sys

def addindex(line):
	line = line.split()
	index = 0
	i = 0
	words = list()
	tags = list()
	while i < len(line):
		if line[i][-1] == ")":
			item = line[i].split(")")
			if item[0] != "":
				tags.append(line[i-1].strip("("))
				words.append(item[0])
				item[0] = item[0]+"-"+str(index)
				line[i] = ")".join(item)
				index += 1
		i += 1
	return " ".join(line),index,words,tags

def extractLabel(span):
	span = span.split()
	if len(span) == 2:
		return "NULL", -1, -1
	label = span[0][1:]
	i = 0
	s = -1
	e = -1
	while i < len(span):
		if span[i][-1] == ")":
			if s == -1:
				s = int(span[i].strip(")").split("-")[-1])
			e = int(span[i].strip(")").split("-")[-1])
		i += 1
	return label,s,e

def getLabel(line,l):
	List = [ [] for i in range(l)]
	stack = []
	for i in range(len(line)):
		if line[i] == "(":
			stack.append(i)
		elif line[i] == ")":
			if len(stack) == 0:
				sys.stderr.write("bracket error\n")
				exit(1)
			#print line[stack[-1]:i+1]
			label,s,e = extractLabel(line[stack[-1]:i+1])
			stack = stack[:-1]
			if label == "NULL" and s == -1 and e == -1:
				continue
			List[s].append("[L1]"+label+"-s")
			List[e].append("[L2]"+label+"-e")
	return List

def process(line):
	line,l,words,tags = addindex(line)
	List = getLabel(line,l)
	for i in range(l):
		print words[i],
	print "|||",
	for i in range(l):
		for item in List[i]:
			if "[L1]" == item[:4]:
				print item[4:],
		print "NULL |||", 
	print
for line in open(sys.argv[1]):
	line = line.strip()
	if line == "":
		continue
	process(line)	
