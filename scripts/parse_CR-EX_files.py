#!/usr/bin/env python3

"""
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
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

import os
import sys

from scipy import stats

if len(sys.argv) < 2:
    print("USAGE: file_matching_pattern")
    print("EXAMPLE: python3 parse_CR-EX_files.py example_directory/*.csv\n")
    exit(-1)

is_EX = False
skiprows = 0
is_GPU = False
is_HOST = False

files = sys.argv[1:]
file_tuples = [] # Tuples are (input name, input size, file object)

# Open all files and get the information for the tuple
if len(files) > 0:
  print("Found", len(files), "files")
  temp = open(files[0])
  
  # Figure out important run-specifc values
  lines = temp.readlines()
  for i in range(len(lines)):
    line = lines[i]
    if line.startswith('algorithm,'):
      skiprows = i
      if 'host' in line:
        is_EX = True
        is_HOST = True
      if 'device' in line:
        is_EX = True
        is_GPU = True
      break
  temp.close()
  
  # Collect the tuple data for each file 
  for file in files:
    curr_file = open(file)
    line = curr_file.readline()
    line = curr_file.readline()
    line = curr_file.readline() # Read 3 lines to get to the input size
    input_size = line.split(',')[1].strip()
    
    # Skip to the acutal data for this file
    curr_row = 3
    # Set these to zero so they can always be added without tracking if there is a preprocessor or not
    first_enc = 0
    first_dec = 0
    second_enc = 0
    second_dec = 0
    while curr_row <= skiprows:
        line = curr_file.readline()
        curr_row = curr_row + 1
        if 'preprocessor encoding time' in line and is_EX:
          # Advance to next
          line = curr_file.readline()
          curr_row = curr_row + 1
          first_enc = float(line.split(',')[0])
          first_dec = float(line.split(',')[1])
          if is_HOST and is_GPU:
            second_enc = float(line.split(',')[2])
            second_dec = float(line.split(',')[3])
            
    file_tuples.append((file, int(input_size), curr_file, first_enc, first_dec, second_enc, second_dec))
else:
  print("ERROR: No files found")
  exit(-1)
  
# The current input columns are [algorithm, compressed bytes, host comp time, host decomp time, device comp time, device decomp time]

writefile = open("all_EX_pipelines_parsed.csv" if is_EX else "all_CR_pipelines_parsed.csv", 'w')
# Header
writefile.write(" , , , , ")
if is_EX:
    if is_GPU and is_HOST:
      writefile.write(" , , , , ")
      writefile.write(" , , , , ")
      writefile.write(" , , , ")
      writefile.write(" , , ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", CR")
      writefile.write(", ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", host c tp")
      writefile.write(", ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", host d tp")
      writefile.write(", ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", dev c tp")
      writefile.write(", ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", dev d tp")
    else:
      ex_type = "host" if is_HOST else "dev"
      writefile.write(" , , , ")
      writefile.write(" , , ")
      writefile.write(" , , ")
      writefile.write(" , ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", CR")
      writefile.write(", ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", " + ex_type + " c tp")
      writefile.write(", ") #spacer
      for i in range(len(file_tuples)):
        writefile.write(", " + ex_type + " d tp")
else:
    writefile.write(", ") #spacer
    for i in range(len(file_tuples)):
        writefile.write(", CR")
writefile.write("\n")
# End header

# Column names
writefile.write("pipeline, hm CR, gm CR, min CR, max CR")
if is_EX:
    if is_GPU and is_HOST:
      writefile.write(", gm host comp tp, gm device comp tp, gm host decomp tp, gm device decomp tp, ")
      writefile.write("min host comp tp, min device comp tp, min host decomp tp, min device decomp tp, ")
      writefile.write("max host comp tp, max device comp tp, max host decomp tp, max device decomp tp")
      writefile.write(", ") #spacer
      for tup in file_tuples:
        writefile.write(", " + tup[0].split('/')[-1])
      writefile.write(", ") #spacer
      for tup in file_tuples:
        writefile.write(", " + tup[0].split('/')[-1])
      writefile.write(", ") #spacer
      for tup in file_tuples:
        writefile.write(", " + tup[0].split('/')[-1])
      writefile.write(", ") #spacer
      for tup in file_tuples:
        writefile.write(", " + tup[0].split('/')[-1])
    else:
      ex_type = "host" if is_HOST else "dev"
      writefile.write(", gm " + ex_type + " comp tp, gm " + ex_type + " decomp tp, ")
      writefile.write("min " + ex_type + " comp tp, min " + ex_type + " decomp tp, ")
      writefile.write("max " + ex_type + " comp tp, max " + ex_type + " decomp tp")
      writefile.write(", ") #spacer
      for tup in file_tuples:
        writefile.write(", " + tup[0].split('/')[-1])
      writefile.write(", ") #spacer
      for tup in file_tuples:
        writefile.write(", " + tup[0].split('/')[-1])
writefile.write(", ") #spacer
for tup in file_tuples:
  writefile.write(", " + tup[0].split('/')[-1])
writefile.write("\n")
# End column names

best_gm = [0, "pipeline"]
best_hm = [0, "pipeline"]

# Actual data, one row at a time until the end of the file
while True:
  pipeline_rows = []
  pipeline = ""
  # Check for end of file
  first_row = file_tuples[0][2].readline()
  # Not at end of file
  if first_row:
    pipeline = first_row.split(',')[0]
    pipeline_rows.append(first_row.strip())
    for i in range(1, len(file_tuples)):
      tup = file_tuples[i]
      pipeline_rows.append(tup[2].readline().strip())
  # At end of file, leave the while loop
  else:
    break
  
  crs = []
  first_ctp = []
  first_dtp = []
  second_ctp = []
  second_dtp = []
  for i in range(len(file_tuples)):
    crs.append(file_tuples[i][1] / int(pipeline_rows[i].split(',')[1]))
    if is_EX:
      first_ctp.append(file_tuples[i][1] / (float(pipeline_rows[i].split(',')[2]) + file_tuples[i][3]))
      first_dtp.append(file_tuples[i][1] / (float(pipeline_rows[i].split(',')[3]) + file_tuples[i][4]))
      if is_GPU and is_HOST:
        second_ctp.append(file_tuples[i][1] / (float(pipeline_rows[i].split(',')[4]) + file_tuples[i][5]))
        second_dtp.append(file_tuples[i][1] / (float(pipeline_rows[i].split(',')[5]) + file_tuples[i][6]))

  pipe_cr_hm = stats.hmean(crs)
  if pipe_cr_hm > best_hm[0]:
    best_hm[0] = pipe_cr_hm
    best_hm[1] = pipeline
  pipe_cr_gm = stats.gmean(crs)
  if pipe_cr_gm > best_gm[0]:
    best_gm[0] = pipe_cr_gm
    best_gm[1] = pipeline
  pipe_first_ctp_gm = -1
  pipe_first_dtp_gm = -1
  pipe_second_ctp_gm = -1
  pipe_second_dtp_gm = -1
  if is_EX:
    pipe_first_ctp_gm = stats.gmean(first_ctp)
    pipe_first_dtp_gm = stats.gmean(first_dtp)
    if is_GPU and is_HOST:
      pipe_second_ctp_gm = stats.gmean(second_ctp)
      pipe_second_dtp_gm = stats.gmean(second_dtp)

  pipe_cr_min = stats.tmin(crs)
  pipe_first_ctp_min = -1
  pipe_first_dtp_min = -1
  pipe_second_ctp_min = -1
  pipe_second_dtp_min = -1
  if is_EX:
    pipe_first_ctp_min = stats.tmin(first_ctp)
    pipe_first_dtp_min = stats.tmin(first_dtp)
    if is_GPU and is_HOST:
      pipe_second_ctp_min = stats.tmin(second_ctp)
      pipe_second_dtp_min = stats.tmin(second_dtp)

  pipe_cr_max = stats.tmax(crs)
  pipe_first_ctp_max = -1
  pipe_first_dtp_max = -1
  pipe_second_ctp_max = -1
  pipe_second_dtp_max = -1
  if is_EX:
    pipe_first_ctp_max = stats.tmax(first_ctp)
    pipe_first_dtp_max = stats.tmax(first_dtp)
    if is_GPU and is_HOST:
      pipe_second_ctp_max = stats.tmax(second_ctp)
      pipe_second_dtp_max = stats.tmax(second_dtp)

  writefile.write(str(pipeline) + ', ' + str(pipe_cr_hm) + ', ' + str(pipe_cr_gm) + ', ' + str(pipe_cr_min) + ', ' + str(pipe_cr_max))
  if is_EX:
      if is_GPU and is_HOST:
        writefile.write(', ' + str(pipe_first_ctp_gm) + ', ' + str(pipe_second_ctp_gm) + ', ' + str(pipe_first_dtp_gm) + ', ' + str(pipe_second_dtp_gm) + ', ')
        writefile.write(str(pipe_first_ctp_min) + ', ' + str(pipe_second_ctp_min) + ', ' + str(pipe_first_dtp_min) + ', ' + str(pipe_second_dtp_min) + ', ')
        writefile.write(str(pipe_first_ctp_max) + ', ' + str(pipe_second_ctp_max) + ', ' + str(pipe_first_dtp_max) + ', ' + str(pipe_second_dtp_max))
      else:
        writefile.write(', ' + str(pipe_first_ctp_gm) + ', ' + str(pipe_first_dtp_gm) + ', ')
        writefile.write(str(pipe_first_ctp_min) + ', ' + str(pipe_first_dtp_min) + ', ')
        writefile.write(str(pipe_first_ctp_max) + ', ' + str(pipe_first_dtp_max))
  
  writefile.write(", ") #spacer
  for cr in crs:
    writefile.write(", " + str(cr))
  if is_EX:
    writefile.write(", ") #spacer
    for host_ctp in first_ctp:
      writefile.write(", " + str(host_ctp))
    writefile.write(", ") #spacer
    for host_dtp in first_dtp:
      writefile.write(", " + str(host_dtp))
    if is_GPU and is_HOST:
      writefile.write(", ") #spacer
      for dev_ctp in second_ctp:
        writefile.write(", " + str(dev_ctp))
      writefile.write(", ") #spacer
      for dev_dtp in second_dtp:
        writefile.write(", " + str(dev_dtp))
  writefile.write("\n")
print("Done writing parse")

print("The best geometric mean was", str(round(best_gm[0], 2)) , "from the pipeline" , best_gm[1])
print("The best harmonic mean was", str(round(best_hm[0], 2)) , "from the pipeline" , best_hm[1])
