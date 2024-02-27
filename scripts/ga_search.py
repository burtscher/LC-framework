#!/usr/bin/env python3

#
# ga_search.py
#
# A script which runs a genetic algorithm (GA) to speed up the search for
# a compression algorithm. This is a work in progress.
#

"""
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2024, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
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

import statistics
import subprocess
import random
import copy
import argparse
import datetime
import sys
import os
import typing

class Algorithm:
    """ A class to represent an individual in the population, an algorithm
        composed of `lc` components.
    """

    next_id = 0

    def __init__(self, stages: int, pipeline_d: typing.Dict[str, float], components=None, ccomponents=None) -> None:
        Algorithm.next_id += 1
        self.myid = Algorithm.next_id
        self.p1id = -1
        self.p2id = -1
        self.stages = stages
        if components and ccomponents:
            self.generate_random_algo(components, ccomponents)
        self.pipeline_d = pipeline_d

    def generate_random_algo(self, components, ccomponents) -> None:
        self.algo = [random.choice(components) for _ in range(self.stages - 1)]
        self.algo.append(random.choice(ccomponents))
        self.algo_str = ' '.join(self.algo)

    def update_algo(self, algo: typing.List[str]) -> None:
        self.algo = algo
        self.update_algo_str()

    def update_algo_str(self) -> None:
        self.algo_str = ' '.join(self.algo)

    def update_id(self) -> None:
        Algorithm.next_id += 1
        self.myid = Algorithm.next_id

    def reset_id(self) -> None:
        Algorithm.next_id = 1

    def check_algorithm(self) -> bool:
        return self.algo_str in self.pipeline_d

    def run_algorithm(self, inputs) -> float:
        pipeline_evaluated = self.check_algorithm()
        if not pipeline_evaluated:
            ratios = []

            # run each file
            for inp in inputs:
                command = self.get_cmd(inp)
                res = subprocess.run(command,
                                     shell=True,
                                     text=True,
                                     capture_output=True)
                ratios.append(parse_comp_ratio(res.stdout, res.stderr))

            # compute the geometric mean
            self.comp_ratio = statistics.geometric_mean(ratios)
            self.pipeline_d[self.algo_str] = self.comp_ratio
        else:
            self.comp_ratio = self.pipeline_d[self.algo_str]

        return self.comp_ratio

    #
    # NOTE: Using a sequence of arguments like the following
    #
    #   ['./lc', 'input_file', 'CR', '""', '"BIT_1"']
    #
    # might cause issues with `lc`.
    #
    # Instead, use a command string with `shell=True' like below:
    #
    # subprocess.run(command,
    #                shell=True,
    #                text=True,
    #                capture_output=True)
    #
    def get_cmd(self, input_file) -> str:
        global preprocessors
        if preprocessors is None:
            return f'./lc {input_file} CR "" "{self.algo_str}"'
        else:
            return f'./lc {input_file} CR "{preprocessors}" "{self.algo_str}"'


class Logger:
    def __init__(self, seed, args) -> None:
        self.args = args
        self.seed = seed
        dt = datetime.datetime.now()
        self.id = f'{dt.date()}-{dt.hour:02}{dt.minute:02}{dt.second:02}'
        self.write_info()

    def write_info(self) -> None:
        path = os.path.join("ga_logs", str(self.id), "_info.txt")

        # make dir, write info in file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f'ID: {self.id}\n')
            f.write(f'Input file: {self.args.inputs}\n')
            f.write(f'Seed: {self.seed}\n')
            f.write(f'Stages: {self.args.stages}\n')
            f.write(f'Generations: {self.args.generations}\n')
            f.write(f'Population size: {self.args.population}\n')
            f.write(f'Elitism cutoff: {self.args.elitism_cutoff}\n')
            f.write(f'Mutation rate: {self.args.mutation_rate}\n')

    def write_population(self, generation: int,
                         population: typing.List[Algorithm]) -> None:
        path = os.path.join("ga_logs", str(self.id),
                            "generations", f'{generation:04}.gen')

        # make dir, write info in file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for p in population:
                f.write(f'{p.comp_ratio}|{p.algo_str}|{p.myid:04}\n')

    def write_event(self, obj_id: int, message: str) -> None:
        path = os.path.join("ga_logs", str(self.id),
                            "individuals", f'{obj_id:04}.indiv')

        # make dir, write info in file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(f'{message}\n')


class Runner:
    def __init__(self, args) -> None:
        # set namespace variable, args
        self.args = args

        # set file name
        global inputs
        inputs = self.args.inputs

        # set preprocessors
        global preprocessors
        preprocessors = self.args.preprocessors

        # set random seed
        random.seed(self.args.randomseed)

        # logger init
        if self.args.logger:
            self.log = Logger(args.randomseed, args)
            if self.args.debug:
                print(f'!Logger initialized')

        # `lc` variables
        self.components, self.components_compress = component_list()
        self.algo_stages = self.args.stages

        # genetic algorithm variables
        self.generation_count = self.args.generations
        self.population_size = self.args.population
        self.elitism_cutoff = self.args.elitism_cutoff
        self.mutation_rate = self.args.mutation_rate

        # verify (might exit the program)
        float_in_range(self.elitism_cutoff, 'elitism cutoff')
        float_in_range(self.mutation_rate, 'mutation_rate')

        self.pipeline_d: dict[str, float]
        self.pipeline_d = {}
        
        self.popul = [Algorithm(self.algo_stages, self.pipeline_d, components=self.components, ccomponents=self.components_compress)
                      for _ in range(self.population_size)]
        self.ratio_history = []

        if self.args.debug:
            print(f'!Runner initialized')

    def run(self) -> None:
        self.ratio_history = []

        if self.args.debug:
            print(f'!Running...')

        for g in range(self.generation_count):
            #
            # evaluation - run each individual's algorithm and sort by fitness
            #
            if self.args.parallel:
                self.call_processes()
                self.popul.sort(reverse=True, key=get_comp_ratio)
            else:
                self.popul.sort(reverse=True, key=run_algo)

            # trim any extra
            while len(self.popul) > self.population_size:
                self.popul.pop()

            best = copy.deepcopy(self.popul[0])
            best.update_id()
            self.ratio_history.append(best.comp_ratio)
            if self.args.logger:
                self.log.write_event(best.myid, f'g {g + 1}: best, {best.algo_str}, {best.comp_ratio}x')

            # print info about this generation
            if not self.args.quiet:
                print(f'~Generation {g}~')
                if self.args.debug:
                    print_population(self.popul)
                else:
                    print_reduced_population(self.popul)
                print('~~~~~\n')

            # write info to disk
            if self.args.logger:
                self.log.write_population(g + 1, self.popul)

            #
            # elitism - keep the best performing in the next generation
            #
            adjusted_fitness = [pow(f.comp_ratio, 3) for f in self.popul]
            fittest = adjusted_fitness[0]
            elite_fitness = [f for f in adjusted_fitness if (1.0 - (f / fittest)) <= self.elitism_cutoff]
            elite_i = len(elite_fitness) + 1

            # the new pop. starts with the elite from this generation
            new_population = self.popul[0:elite_i]

            if self.args.debug:
                print(f'ELITE')
                print_population(new_population)
                print()

            # log
            if self.args.logger:
                for p in new_population:
                    self.log.write_event(p.myid, f'g {g + 1}: elite, {p.algo_str}, {p.comp_ratio}x')

            #
            # selection & crossover - select two parents and do crossover to create
            # two children to add to the new population
            #
            selection_pool = self.popul
            weights = [pow(p.comp_ratio, 3) for p in selection_pool]
            while len(new_population) < self.population_size:
                if self.args.SELECTION_METHOD == 'roulette_wheel':
                    p1, p2 = roulette_wheel_selection(selection_pool, weights)
                else:
                    p1, p2 = tournament_selection(selection_pool, 3)

                if self.args.CROSSOVER_METHOD == 'masked':
                    c1, c2 = masked_crossover(p1, p2)
                else:
                    c1, c2 = single_point_crossover(p1, p2)

                new_population.append(c1)
                new_population.append(c2)
                if self.args.logger:
                    self.log.write_event(c1.myid, f'g {g + 1}: birth, {c1.algo_str}\n\tParent IDs: {c1.p1id}, {c1.p2id}')
                    self.log.write_event(c2.myid, f'g {g + 1}: birth, {c2.algo_str}\n\tParent IDs: {c2.p1id}, {c2.p2id}')

            self.popul = new_population

            #
            # mutation - mutate the whole population (except the best)
            #
            self.popul.append(best)

            # Each component has an equal chance to mutate
            mu = self.mutation_rate
            for p_i in range(len(self.popul) - 1):
                for c_i in range(self.popul[p_i].stages):
                    if random.random() < mu:
                        if c_i == (self.popul[p_i].stages - 1):
                            self.popul[p_i].algo[c_i] = random.choice(self.components_compress)
                        else:
                            self.popul[p_i].algo[c_i] = random.choice(self.components)
                        self.popul[p_i].update_algo_str()

                        if self.args.logger:
                            self.log.write_event(self.popul[p_i].myid, f'g {g + 1}: mutate, {self.popul[p_i].algo_str}')

        if self.args.parallel:
            self.call_processes()
            self.popul.sort(reverse=True, key=get_comp_ratio)
        else:
            self.popul.sort(reverse=True, key=run_algo)

        # print info about the final generation
        if not self.args.quiet:
            print(f'~Generation {self.generation_count}~')
            if self.args.debug:
                print_population(self.popul)
            else:
                print_reduced_population(self.popul)
            print('~~~~~\n')

        self.best_algo_str = self.popul[0].algo_str
        self.best_comp_ratio = self.popul[0].comp_ratio
        self.ratio_history.append(self.best_comp_ratio)
        print(f'Best Algorithm Found: {self.best_algo_str}')
        print(f'Compression Ratio: {self.best_comp_ratio}')
        if self.args.logger:
            print(f'Individual ID: {self.popul[0].myid}')

    def call_processes(self) -> None:
        # parallelize running each individual's algorithm
        # https://stackoverflow.com/questions/14533458/python-threading-multiple-bash-subprocesses
        MAX_PROC = 15

        local_popul = []
        for p in self.popul:
            pipeline_eval = p.check_algorithm()
            if not pipeline_eval:
                local_popul.append(p)
            else:
                p.comp_ratio = p.pipeline_d[p.algo_str]

        ratios_list: list[list[float]]
        ratios_list = [[] for _ in local_popul]
        for inp in inputs:
            cmds = [e.get_cmd(inp) for e in local_popul]
            processes = []

            evaluated = 0
            while evaluated < len(cmds):
                if len(cmds) - evaluated < MAX_PROC:
                    local_cmds = cmds[evaluated:]
                    local_processes = [subprocess.Popen(
                        cmd, shell=True, stdout=subprocess.PIPE, text=True) for cmd in local_cmds]
                    evaluated = len(cmds)

                    for p in local_processes:
                        p.wait()

                    processes += local_processes

                else:
                    local_cmds = cmds[evaluated:MAX_PROC+evaluated]
                    local_processes = [subprocess.Popen(
                        cmd, shell=True, stdout=subprocess.PIPE, text=True) for cmd in local_cmds]
                    evaluated += MAX_PROC

                    for p in local_processes:
                        p.wait()

                    processes += local_processes

            for i, p in enumerate(local_popul):
                stdout, stderr = processes[i].communicate()
                ratios_list[i].append(parse_comp_ratio(stdout, stderr))
        
        member: Algorithm
        for i, member in enumerate(local_popul):
            member.comp_ratio = statistics.geometric_mean(ratios_list[i])
            member.pipeline_d[member.algo_str] = member.comp_ratio


def roulette_wheel_selection(pop: typing.List[Algorithm], weights: typing.List[float]) -> typing.Tuple[Algorithm, Algorithm]:
    parents = random.choices(pop, weights=weights, k=2)
    return (parents[0], parents[1])


def tournament_selection(pop: typing.List[Algorithm], k: int) -> typing.Tuple[Algorithm, Algorithm]:
    picks = random.sample(pop, k=k)
    picks.sort(reverse=True, key=get_comp_ratio)
    p1 = picks[0]
    picks = random.sample(pop, k=k)
    picks.sort(reverse=True, key=get_comp_ratio)
    p2 = picks[0]

    return (p1, p2)


def run_algo(e) -> float:
    return e.run_algorithm(inputs)


def get_comp_ratio(e: Algorithm) -> float:
    return e.comp_ratio


def parse_comp_ratio(output, err) -> float:
    (_, _, comp_ratio) = output.partition("compression:")
    (_, _, comp_ratio) = comp_ratio.partition("%")
    (comp_ratio, _, _) = comp_ratio.partition("x")
    comp_ratio = comp_ratio.strip()
    comp_ratio = comp_ratio.removesuffix('x')
    try:
        comp_ratio = float(comp_ratio)
    except ValueError:
        print('!ERROR running LC!')
        print('LC Output:')
        print(output)
        print('LC Error:')
        print(err)
        quit()
    return comp_ratio


def compress_only(c: str):
    return c.startswith("R") or c.startswith("C") or c.startswith("H")


def component_list() -> typing.Tuple[typing.List[str], typing.List[str]]:
    res = subprocess.run('./lc', shell=True, text=True, capture_output=True)
    (_, _, components) = res.stdout.partition("available components:")
    components = components.splitlines()
    components = components[1]
    components = components.split()
    components.remove("NUL")
    components_compress = list(filter(compress_only, components))

    return (components, components_compress)


def single_point_crossover(p1: Algorithm, p2: Algorithm) -> typing.Tuple[Algorithm, Algorithm]:
    crossover_point = random.randrange(p1.stages)
    c1 = Algorithm(p1.stages, p1.pipeline_d)
    c2 = Algorithm(p1.stages, p1.pipeline_d)
    a1 = p1.algo[0:crossover_point] + p2.algo[crossover_point:]
    a2 = p2.algo[0:crossover_point] + p1.algo[crossover_point:]

    c1.update_algo(a1)
    c1.update_id()
    c1.p1id = p1.myid
    c1.p2id = p2.myid

    c2.update_algo(a2)
    c2.update_id()
    c2.p1id = p1.myid
    c2.p2id = p2.myid
    return (c1, c2)


def masked_crossover(p1: Algorithm, p2: Algorithm) -> typing.Tuple[Algorithm, Algorithm]:
    mask = get_bitmask_str(p1.stages)
    c1 = Algorithm(p1.stages, p1.pipeline_d)
    c2 = Algorithm(p1.stages, p1.pipeline_d)
    a1 = []
    a2 = []

    for i in range(len(mask)):
        bit = mask[i]
        if bit == '1':  # c1 inherits from p1
            a1.append(p1.algo[i])
            a2.append(p2.algo[i])
        elif bit == '0':
            a1.append(p2.algo[i])
            a2.append(p1.algo[i])

    c1.update_algo(a1)
    c1.update_id()
    c1.p1id = p1.myid
    c1.p2id = p2.myid

    c2.update_algo(a2)
    c2.update_id()
    c2.p1id = p1.myid
    c2.p2id = p2.myid

    return (c1, c2)


def get_bitmask_str(k) -> str:
    mask = random.getrandbits(k)
    mask_str = bin(mask)
    mask_str = mask_str[2:]
    mask_str = mask_str.rjust(k, '0')
    return mask_str


def print_population(population: typing.List[Algorithm]) -> None:
    print(f'Compression Ratio\tComponent List')
    for p in population:
        print(f'{p.comp_ratio}\t\t\t{p.algo_str}')


def print_reduced_population(population: typing.List[Algorithm]) -> None:
    if len(population) < 6:
        print_population(population)
        return

    print(f'Compression Ratio\tComponent List')
    print(f'-Top 3-')
    print(f'{population[0].comp_ratio}\t\t\t{population[0].algo_str}')
    print(f'{population[1].comp_ratio}\t\t\t{population[1].algo_str}')
    print(f'{population[2].comp_ratio}\t\t\t{population[2].algo_str}')
    print(f'-Bottom 3-')
    print(f'{population[-3].comp_ratio}\t\t\t{population[-3].algo_str}')
    print(f'{population[-2].comp_ratio}\t\t\t{population[-2].algo_str}')
    print(f'{population[-1].comp_ratio}\t\t\t{population[-1].algo_str}')


def float_in_range(x: float, s: str) -> None:
    if 0.0 <= x <= 1.0:
        return
    print(f'ERROR: {s} not in range [0.0, 1.0] !')
    sys.exit(1)


def main() -> None:
    # parser init
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs='+',
        help="path(s) to the file you want to compress")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-d", "--debug",
        help="print debug statements to the console",
        action="store_true")
    group.add_argument(
        "-q", "--quiet",
        help="silence output during computation",
        action="store_true")
    parser.add_argument(
        "-l", "--logger",
        help="write logs to dir named 'ga_logs'",
        action="store_true")
    parser.add_argument(
        "--parallel",
        help="create a process per individual for computing fitness, lc should be compiled without -fopenmp, may cause issues with large files",
        action="store_true")
    parser.add_argument(
        "-r", "--randomseed",
        help="number that seeds the RNG",
        action="store",
        type=int)
    parser.add_argument(
        "-s", "--stages",
        help="number of stages in the algorithm",
        action="store",
        default=3, type=int)
    parser.add_argument(
        "-g", "--generations",
        help="number of generations to run",
        action="store",
        default=16, type=int)
    parser.add_argument(
        "-p", "--population",
        help="number of individuals in the population",
        action="store",
        default=16, type=int)
    parser.add_argument(
        "-e", "--elitism_cutoff",
        help="fitness of each individual has to be within this percentage of the fittest to live on to the next generation",
        action="store",
        default=0.1, type=float)
    parser.add_argument(
        "-m", "--mutation_rate",
        help="probability that a component will be mutated",
        action="store",
        default=0.5, type=float)
    parser.add_argument(
        '--SELECTION_METHOD', help='the selection method, default is roulette_wheel',
        choices=['roulette_wheel', 'tournament'], default='tournament')
    parser.add_argument(
        '--CROSSOVER_METHOD', help='the crossover method, default is masked',
        choices=['single_point', 'masked'], default='masked')
    parser.add_argument(
        '-o', '--preprocessors',
        help='preprocessor argument passed to LC')

    # collect args
    args = parser.parse_args()

    # set random seed if none is given
    if args.randomseed is None:
        args.randomseed = random.randrange(sys.maxsize)

    if args.debug:
        print(f'!Executing with the following arguments: {vars(args)}')

    if not args.quiet:
        print(f'~Parameters~')
        print(f'Input file(s): {args.inputs}')
        print(f'Stages: {args.stages}')
        print(f'Generations: {args.generations}')
        print(f'Population size: {args.population}')
        print(f'Elitism cutoff: {args.elitism_cutoff}')
        print(f'Mutation rate: {args.mutation_rate}')
        print(f'Selection method: {args.SELECTION_METHOD}')
        print(f'Crossover method: {args.CROSSOVER_METHOD}')
        print(f'Random seed: {args.randomseed}')
        print()

    # create runner
    r = Runner(args)
    r.run()


if __name__ == "__main__":
    main()
