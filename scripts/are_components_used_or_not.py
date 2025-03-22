#!/usr/bin/python3

# The goal of this script is to read in csv files produced by LC running in EX mode
# and produce a list of compression components used in the top X percent (specified
# at command line) of algorithms. For the purposes of this file, the "best" algorithm
# has the best compression.

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

import sys
import os
import re
import csv

# get length of command line
n = len(sys.argv)
if n < 2:
    print("USAGE: percent_tolerance_to_top_compression_ratio file_matching_pattern")
    print("EXAMPLE: python3 are_stages_used_or_not.py 10 ./example_directory/*.csv\n")
    sys.exit()

percent_tolerance = float(sys.argv[1])
print("Tolerance: ", percent_tolerance, "%")
percent_tolerance = percent_tolerance/100.0

files = sys.argv[2:]

if len(files) > 0:
    # verify that there are .csv files (besides summary_file.csv)
    print("Found", len(files), "files")

    is_csv = False
    used_or_not_count = 0
    for filename in files:
        if re.search("\\.csv", filename) and not re.search("used_or_not.*csv", filename):
            is_csv = True
            break
    if is_csv == False:
        print("ERROR: No .csv files found\n")
        sys.exit()

    filename_not_found = True
    filename_number = -1
    temp_filename = "used_or_not_" + str(filename_number) + ".csv"
    while filename_not_found:
        filename_number = filename_number + 1
        temp_filename = "used_or_not_" + str(filename_number) + ".csv"
        filename_not_found = False
        for filename in os.listdir(os.getcwd()):
            if re.search(temp_filename, filename):
                filename_not_found = True
                break

    summaryfilename = temp_filename
    used_components = set()
    unused_components = set()
    used_components_per_stage = [set() for i in range(8)]  # LC has a max of 8 stages
    unused_components_per_stage = [set() for i in range(8)]
    all_components = set()

    # first gotta create the summary csv file
    with open(summaryfilename, mode='w', newline='') as summary_file:
        summary_writer = csv.writer(summary_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        summary_writer.writerow(
            ["input file", "compressed size [bytes]", "best alg"])

        # loop and read all files in working_directory, parse their data, and write info to new csv
        for filename in files:
            if re.search("\\.csv", filename) and not re.search("used_or_not.*csv", filename):
                # file specific variables
                line_count = 0
                best_comp_size = sys.maxsize
                best_alg = "ERROR"
                input_name = "ERROR"
                simple_input_name = "ERROR"

                # need this to determine where the data begins (works for different inputs)
                data_started = False
                available_components = 0

                # open file for reading
                with open(filename) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        if line_count == 1:
                            # grab input file name
                            input_name = row[1]
                            print("\n" + input_name + "\n")
                            temporary = re.search('\/([a-zA-Z0-9_-]+)\.', row[1])
                            if temporary:
                                simple_input_name = temporary.group(1)
                                print("\n" + simple_input_name + "\n")
                            else:
                                simple_input_name = input_name
                                print("WARNING: Unusual filename ", simple_input_name, ", double check run command")
                        elif data_started:
                            # check for minimum compression value
                            if int(row[1]) < best_comp_size:
                                best_comp_size = int(row[1])
                                best_alg = row[0]
                        elif available_components != 0:
                            available_components -= 1
                            temp = row[0].split()
                            for t in temp:
                                all_components.add(t)
                        try:
                            if row[0] == "available components":
                                available_components = int(row[1])
                        except:
                            pass

                        line_count += 1
                        # check if next row is where the data starts
                        try:
                            if row[1] == " compressed size [bytes]":
                                data_started = True
                        except:
                            pass

                    print(f'Processed {line_count} lines.')
                    print(f'Best compression size: {best_comp_size}')

                    summary_writer.writerow(
                        [simple_input_name, best_comp_size, best_alg])
                    # summary_writer.writerow([""])

                    data_started = False
                    upper_bound = best_comp_size + (percent_tolerance * float(best_comp_size))
                    upper_bound = int(upper_bound)
                    print("Upper bound of compressed size[bytes]: ", upper_bound)

                with open(filename) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        if data_started:
                            if int(row[1]) <= upper_bound:
                                temp = row[0].split()
                                stage = 0
                                for t in temp:
                                    used_components.add(t)
                                    used_components_per_stage[stage].add(t)
                                    stage = stage + 1
                        try:
                            if row[1] == " compressed size [bytes]":
                                data_started = True
                        except:
                            pass

        used_run_string = "\""
        unused_run_string = "\"~"
        for component in all_components:
            if component not in used_components:
                unused_components.add(component)
            for i in range(8):
                if component not in used_components_per_stage[i]:
                    unused_components_per_stage[i].add(component)
        summary_writer.writerow([""])
        summary_writer.writerow(["Used components:"])
        print("\nUsed Components:")
        for component in sorted(used_components):
            summary_writer.writerow([component])
            print(component)
            used_run_string += component + "|"
        used_run_string = used_run_string[:-1] + "\""
        summary_writer.writerow([used_run_string])
        # print("\nRun with:")
        print(used_run_string)

        summary_writer.writerow([""])
        summary_writer.writerow(["Unused components:"])
        print("\nUnused Components:")
        for component in sorted(unused_components):
            summary_writer.writerow([component])
            print(component)
            unused_run_string += component + "|"
        unused_run_string = unused_run_string[:-1] + "\""
        summary_writer.writerow([unused_run_string])
        # print("\nRun with:")
        print(unused_run_string)

        summary_writer.writerow([""])
        summary_writer.writerow([""])
        for i in range(8):
            if not len(used_components_per_stage[i]) == 0:
                used_components_per_stage_run_string = "\""
                unused_components_per_stage_run_string = "\"~"
                
                summary_writer.writerow(["Stage " + str(i+1) + " Used Components:"])
                print("\nStage " + str(i+1) + " Used Components:")
                for component in sorted(used_components_per_stage[i]):
                    summary_writer.writerow([component])
                    print(component)
                    used_components_per_stage_run_string += component + "|"
                used_components_per_stage_run_string = used_components_per_stage_run_string[:-1] + "\""
                summary_writer.writerow([used_components_per_stage_run_string])
                print(used_components_per_stage_run_string)
                summary_writer.writerow([""])

                summary_writer.writerow(["Stage " + str(i + 1) + " Unused Components:"])
                print("\nStage " + str(i + 1) + " Unused Components:")
                for component in sorted(unused_components_per_stage[i]):
                    summary_writer.writerow([component])
                    print(component)
                    unused_components_per_stage_run_string += component + "|"
                unused_components_per_stage_run_string = unused_components_per_stage_run_string[:-1] + "\""
                summary_writer.writerow([unused_components_per_stage_run_string])
                print(unused_components_per_stage_run_string)
                summary_writer.writerow([""])

else:
    print("ERROR: NO FILES SPECIFIED\n")
sys.exit()


