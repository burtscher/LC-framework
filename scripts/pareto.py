#!/usr/bin/env python3

"""
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2023, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
"""

import csv
import functools
import os

class Elem:
    def __init__(self, pipe, CR, HencThru, HdecThru, DencThru, DdecThru):
        self.pipe = pipe
        self.CR = CR
        self.HencThru = HencThru
        self.HdecThru = HdecThru
        self.DencThru = DencThru
        self.DdecThru = DdecThru

def compareElemHencThru(e1, e2):
    if e1.CR == e2.CR:
      if e1.HencThru < e2.HencThru:
          return -1
      else:
          return 1
    if e1.CR < e2.CR:
        return -1
    else:
        return 1

def compareElemHdecThru(e1, e2):
    if e1.CR == e2.CR:
      if e1.HdecThru < e2.HdecThru:
          return -1
      else:
          return 1
    if e1.CR < e2.CR:
        return -1
    else:
        return 1


def compareElemDencThru(e1, e2):
    if e1.CR == e2.CR:
      if e1.DencThru < e2.DencThru:
          return -1
      else:
          return 1
    if e1.CR < e2.CR:
        return -1
    else:
        return 1

def compareElemDdecThru(e1, e2):
    if e1.CR == e2.CR:
      if e1.DdecThru < e2.DdecThru:
          return -1
      else:
          return 1
    if e1.CR < e2.CR:
        return -1
    else:
        return 1


filename = 'all_EX_pipelines_parsed.csv'

if not os.path.exists(filename):
    print(f"Error: {filename} not found!")
else:
    with open(filename, 'r') as f:
        reader = csv.reader(f)
    next(reader) # Skip first row
    reader = csv.reader(f)
    next(reader) # Skip second row
    reader = csv.reader(f)
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    elems = []
    for row in reader:
        col1.append(row[0])
        col2.append(float(row[1]))
        col3.append(float(row[2]))
        col4.append(float(row[3]))
        col5.append(float(row[4]))
        col6.append(float(row[5]))
        col7.append(float(row[6]))
        elems.append(Elem(row[0], float(row[1]), float(row[4]), float(row[6]), float(row[5]), float(row[7])))

print("Pareto host encoding")
print("--------------------      *        *")

elems = sorted(elems, key=functools.cmp_to_key(compareElemHencThru))

for i in range(len(elems)):
  ok = True
  if elems[i].CR <= 1.0:
    ok = False
  else:
    for j in range(i + 1, len(elems)):
      if elems[i].HencThru < elems[j].HencThru:
        ok = False
        break
  if ok:
    print("{:<21s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(elems[i].pipe, elems[i].CR, elems[i].HencThru / 1000000000.0, elems[i].HdecThru / 1000000000.0, elems[i].DencThru / 1000000000.0, elems[i].DdecThru / 1000000000.0))



print("\nPareto host decoding")
print("--------------------      *              *")

elems = sorted(elems, key=functools.cmp_to_key(compareElemHdecThru))

for i in range(len(elems)):
  ok = True
  if elems[i].CR <= 1.0:
    ok = False
  else:
    for j in range(i + 1, len(elems)):
      if elems[i].HdecThru < elems[j].HdecThru:
        ok = False
        break
  if ok:
    print("{:<21s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(elems[i].pipe, elems[i].CR, elems[i].HencThru / 1000000000.0, elems[i].HdecThru / 1000000000.0, elems[i].DencThru / 1000000000.0, elems[i].DdecThru / 1000000000.0))



print("\nPareto Device encoding")
print("--------------------      *                      *")

elems = sorted(elems, key=functools.cmp_to_key(compareElemDencThru))

for i in range(len(elems)):
  ok = True
  if elems[i].CR <= 1.0:
    ok = False
  else:
    for j in range(i + 1, len(elems)):
      if elems[i].DencThru < elems[j].DencThru:
        ok = False
        break
  if ok:
    print("{:<21s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(elems[i].pipe, elems[i].CR, elems[i].HencThru / 1000000000.0, elems[i].HdecThru / 1000000000.0, elems[i].DencThru / 1000000000.0, elems[i].DdecThru / 1000000000.0))



print("\nPareto Device decoding")
print("--------------------      *                              *")

elems = sorted(elems, key=functools.cmp_to_key(compareElemDdecThru))

for i in range(len(elems)):
  ok = True
  if elems[i].CR <= 1.0:
    ok = False
  else:
    for j in range(i + 1, len(elems)):
      if elems[i].DdecThru < elems[j].DdecThru:
        ok = False
        break
  if ok:
    print("{:<21s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}".format(elems[i].pipe, elems[i].CR, elems[i].HencThru / 1000000000.0, elems[i].HdecThru / 1000000000.0, elems[i].DencThru / 1000000000.0, elems[i].DdecThru / 1000000000.0))
