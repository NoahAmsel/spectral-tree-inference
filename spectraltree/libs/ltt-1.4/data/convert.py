"""Tool to convert UCI dataset to matlab readable format"""

__author__ = "Stefan Harmeling"

# created: 07JAN2008

# run this script like this:
#
#   python convert.py car.data
#
# will generate
#
#   car.data.numeric
#
# description:
#
#   convert comma-separated data of strings into comma-separated data
#   of integers

from string import split
from sys import argv

fname = argv[-1]

# read the original data file
f = open(fname)
data = f.readlines()
f.close()

# chop off the newlines
def chomp(line):
    if line[-1]=='\n':
        return line[:-1]
    else:
        return line
data = map(chomp, data)  # remove possible newlines

# filter out empty lines
data = filter(lambda line: line!="", data)

# split each line at each comma
def splitcomma(line):
    return split(line, ",")
data = map(splitcomma, data) # split at each comma

# use a list of dictionaries to generate the new integer keys
nfeatures = len(data[0])
codes = []
for i in range(nfeatures):
    codes.append({})

# process each data point separately
for datum in data:
    if len(datum) != nfeatures:
        error("not all vectors have same length (whitespace?)")
    for i in range(nfeatures):
        key = datum[i]
        code = codes[i]
        if not code.has_key(key):
            code[key] = len(code) + 1  # starts with 1
        datum[i] = code[key]

# write out converted data
f = open(fname+'.numeric', 'w')
# first write out the codes for each feature
for i in range(nfeatures):
    f.write("% feature " + str(i+1) + " " + str(codes[i]) + "\n")
# secondly write out the data
for datum in data:
    line = str(datum)
    f.write(line[1:-1] + "\n")
f.close()

# that's it!

