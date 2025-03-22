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

import re
import glob, os
from pathlib import Path
from os.path import exists
import math
import shutil
import sys
import csv
import subprocess
import re
from subprocess import call


# Check if "lc" is in the current directory
lc_path = "./lc"
if not os.path.isfile(lc_path):
    # Check if "lc" is in the "../" directory
    lc_path = os.path.join("./..", "lc")
    if not os.path.isfile(lc_path):
        # If not found in either location, ask the user for the correct path
        user_input = input("The 'lc' executable was not found in the current directory or '../'. Please provide the correct path to 'lc': ")
        if os.path.isfile(user_input):
            lc_path = user_input
        else:
            print("The provided path is not valid.")


# Name of the script you want to call
script_name = "parse_CR-EX_files.py"

# Check if the script exists in the same directory
if os.path.isfile(script_name):
    script_path = script_name
else:
    # Check if the script exists in a "scripts" folder
    scripts_dir = "scripts"
    script_path = os.path.join(scripts_dir, script_name)

    if not os.path.isfile(script_path):
        # The script was not found in either location, ask the user for the correct directory
        user_input = input(f"The script '{script_name}' was not found in the current directory or '{scripts_dir}' folder. Please provide the correct path to the script: ")
        if os.path.isfile(user_input):
            script_path = user_input
        else:
            print("The provided path is not valid.")




# read components from user input
if len(sys.argv) < 5:
  print("Usage: {} preprocessor_name min_stages max_stages file1 file2 ... fileN".format(sys.argv[0]))  
  # Run the command and capture its output
  try:
      output = subprocess.check_output([lc_path], stderr=subprocess.STDOUT, universal_newlines=True)
  except subprocess.CalledProcessError as e:
      output = e.output

  # Split the output into lines
  lines = output.split('\n')

  # Find the line that contains "available preprocessors"
  preprocessors_line = next((line for line in lines if "available preprocessors" in line.lower()), None)

  # If the preprocessors line is found, extract the preprocessors
  if preprocessors_line:
      preprocessors_start = lines.index(preprocessors_line)
      preprocessors = []
      for line in lines[preprocessors_start + 1:]:
          if re.match(r'\w', line):  # Check if the line starts with a word character
              preprocessors.append(line.strip())
          else:
              break

      # Print the preprocessors
      print("\navailable preprocessors are:\n")
      print('\n'.join(preprocessors))
  else:
      print("Preprocessors information not found in the output.")
  sys.exit() 
else:
  input_strings = sys.argv[1:]
  inp_files = sys.argv[4:]

pre = input_strings[0]
mins = input_strings[1] 
maxs = input_strings[2]

csv_path = "*csv"  

first = "~NUL"
for _ in range(int(mins) - 1):
  first = first + " ~NUL"

print("min", mins)
print("max", maxs)


for i in range(int(maxs) - int(mins) + 1):
  for filename in inp_files:       
          filepath = filename
          pass_arg = []
          pass_arg.append(lc_path)
          pass_arg.append(filepath)
          pass_arg.append("CR")
          pass_arg.append(pre)
          pass_arg.append(first)
          print(pass_arg)
          subprocess.check_call(pass_arg)
      
  files = glob.glob(csv_path)
  print(files)
  call(["python3", script_path] + files)
  with open('all_CR_pipelines_parsed.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip first row
    reader = csv.reader(f)
    next(reader) # Skip second row
    reader = csv.reader(f)
    col1 = []
    col2 = []
    for row in reader:
      col1.append(row[0])
      col2.append(float(row[1]))
    min_value = float('-inf')
    max_index = None
    for k in range(len(col2)):
        if col2[k] > min_value:
            min_value = col2[k]
            max_index = k
    pipe_string = col1[max_index]
    print("best pipeline is: ", pipe_string)
    string_to_remove = "~NUL"

    # Split the original string by spaces
    parts = first.split()

    # Filter out the part you want to remove
    filtered_parts = [part for part in parts if part != string_to_remove]

    # Join the filtered parts back into a string with spaces
    first = " ".join(filtered_parts)
    first =  pipe_string + " ~NUL"
    print("next to try is: ", first)

    csvfiles = glob.glob('*csv')
    for f in csvfiles:
      os.remove(f)
