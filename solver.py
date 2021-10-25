import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from util import *


### Initialization Functions ###

def init_dist_matrix(tsp_file):
    tsp_coordinate_list = read_tsp(tsp_file)
    # Iniitlaize distance matrix
    n_cities, dim = tsp_coordinate_list.shape
    diff_matrix = np.tile(tsp_coordinate_list, (1,n_cities)).reshape((n_cities,n_cities,dim)) - np.tile(tsp_coordinate_list, (n_cities,1)).reshape((n_cities,n_cities,dim))
    dist_matrix = np.linalg.norm(diff_matrix, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)

    return dist_matrix, tsp_coordinate_list

def init_pheromone_matrix(dist_matrix, n_cities):
    # Initialize pheromone matrix
    nn_tour = get_nn_tour(np.copy(dist_matrix), n_cities)
    nn_dist = get_tour_dist(dist_matrix, nn_tour[:-1])
    init_pheromone = 1/(nn_dist*n_cities)

    pheromone_matrix = np.full((n_cities,n_cities), init_pheromone)
    return pheromone_matrix, init_pheromone

def init_ACS_params(args):
    return args.exploit_tendency, args.beta, args.rho, args.alpha

def init_testing_params(args):
    return args.verbose, args.get_stats, args.graph

def init_ants(n_ants, n_cities):
    visited = np.zeros((n_ants, n_cities), dtype=int)
    tours = np.zeros((n_ants,1), dtype=int)
    tour_dists = np.zeros(n_ants)

    # Randomly initialize start position of each ant
    for i in range(n_ants):
        init_city = np.random.choice(n_cities)
        visited[i,init_city] = 1
        tours[i,0] = init_city
        
    return visited, tours, tour_dists

def init_neighbour_matrix(dist_matrix, n_cities):
    neighbour_matrix = np.argsort(dist_matrix)[:,:-1]
    neighbour_matrix = neighbour_matrix[:,:20]
    return neighbour_matrix


### Solver Class ###

class Solver():
    def __init__(self, args):
        self.dist_matrix, self.coord_list = init_dist_matrix(args.tsp_file)
        self.n_cities = self.dist_matrix.shape[0]
        self.best_dist = np.inf
        self.best_tour = None
        self.max_fitness_evals = args.max_fitness_evals
        self.fitness_evals = 0
        self.max_iters = args.max_iters
        self.iters = 0
        self.alg = args.algorithm
        self.verbose, self.stats, self.graph = init_testing_params(args)

        # Initialize variables common to all ant colony systems
        if self.alg in ['ACS', 'AACS', 'HAACS']:
            self.n_ants = args.population
            self.pheromone_matrix, self.init_pheromone = init_pheromone_matrix(self.dist_matrix, self.n_cities)
            self.visited, self.tours, self.tour_dists = init_ants(self.n_ants, self.n_cities)
            self.q0, self.b, self.p, self.a = init_ACS_params(args)

        # Initialize variables common to adaptive ant colony systems
        if self.alg in ['AACS', 'HAACS']:
            self.no_improvement_count = 0

        # Initialize neighbour matrix used for speeding up 2opt searches
        if self.alg in ['2opt', 'HAACS']:
            self.neighbour_matrix = init_neighbour_matrix(self.dist_matrix, self.n_cities)

        # Initialize variables only for HAACS
        if self.alg == 'HAACS':
            self.two_opt_best_dist = np.inf
            self.two_opt_best_tour = None
            self.n_2opt_iters = args.n_2opt_iters
            self.count_2opt_fitness = args.count_2opt_fitness

        if self.verbose:
            self.print_configs(args)
            print('\nSolver initialized', end='\n\n')

    def init_stats(self):
        self.nATS = [] # normalized average tour simiiarity between ants and best tour
        self.best_dist_list = []

    def clear_ant_data(self):
        self.tour_dists = np.zeros(self.n_ants)
        
        # Ants always start from their initialized position
        home_cities = self.tours[:, 0]
        self.tours = np.zeros((self.n_ants,1), dtype=int)
        self.tours[:,0] = home_cities
        self.visited = np.zeros((self.n_ants, self.n_cities), dtype=int)
        self.visited[np.arange(self.n_ants), home_cities] = 1    

    def choose_next_city(self):
        next_cities = np.zeros(self.n_ants, dtype=int)
        next_tour_dist = np.zeros(self.n_ants)

        # probabilistically decide whether to exploit or explore for each ant
        q = np.random.rand(self.n_ants)
        exploit = q <= self.q0

        # calculate ___ (insert appropriate wording)
        choice_matrix = np.multiply(np.power(1/self.dist_matrix, self.b), self.pheromone_matrix)
        for i in range(self.n_ants):
            cur = self.tours[i,-1]
            choice_matrix_copy = np.copy(choice_matrix[cur])

            # set probability of visiting already visited cities to zero
            choice_matrix_copy[self.visited[i] == 1] = 0

            # exploit or explore
            if exploit[i]:
                next = np.argmax(choice_matrix_copy)
            else:
                choice_matrix_copy = choice_matrix_copy / sum(choice_matrix_copy)
                next = np.random.choice(self.n_cities, p=choice_matrix_copy)

            self.visited[i, next] = 1
            next_cities[i] = next
            next_tour_dist[i] = self.dist_matrix[cur, next]
        
        # update tours and tour distances for each ant
        self.tours = np.append(self.tours, next_cities.reshape(self.n_ants,1), axis=1)
        self.tour_dists += next_tour_dist

    def return_to_start(self):
        next_tour_dist = np.zeros(self.n_ants)
        for i in range(self.n_ants):
            cur = self.tours[i,-1]
            next = self.tours[i,0]
            next_tour_dist[i] = self.dist_matrix[cur,next]
        self.tours = np.append(self.tours, self.tours[:,0].reshape(self.n_ants,1), axis=1)
        self.tour_dists += next_tour_dist


    def local_update(self):
        prev_two_cities = self.tours[:, -2:]
        for prev, cur in prev_two_cities:
            self.pheromone_matrix[prev, cur] = (1-self.p) * self.pheromone_matrix[prev, cur] + self.p * self.init_pheromone
            self.pheromone_matrix[cur, prev] = (1-self.p) * self.pheromone_matrix[cur, prev] + self.p * self.init_pheromone

    def find_best_ant(self, adaptive=True):
        # Fitness evaluation (for each ant)
        local_best_ant = np.argmin(self.tour_dists)
        self.fitness_evals += self.n_ants

        if self.tour_dists[local_best_ant] < self.best_dist:
            self.best_dist = self.tour_dists[local_best_ant]
            self.best_tour = self.tours[local_best_ant,:]
        elif adaptive:
            self.no_improvement_count += 1
        
    def global_update(self):
        if self.stats:
            self.best_dist_list.append(self.best_dist)

        # Perform global-best update
        new_pheromone = 1/self.best_dist
        for i in range(self.n_cities):
            prev = self.best_tour[i]
            cur = self.best_tour[i+1]
            self.pheromone_matrix[prev, cur] = (1-self.a) * self.pheromone_matrix[prev, cur] + self.p * new_pheromone
            self.pheromone_matrix[cur, prev] = (1-self.a) * self.pheromone_matrix[cur, prev] + self.p * new_pheromone

    def normalized_average_tour_similarity(self):
        sum_common_edge_count = 0
        best_edge_set = get_edge_set(self.best_tour[:-1])
        for i in range(self.n_ants):
            edge_set = get_edge_set(self.tours[i,:-1])            
            common_edge_count = get_common_edge_count(best_edge_set, edge_set)
            sum_common_edge_count += common_edge_count
        ATS = sum_common_edge_count/self.n_ants
        normalized_ATS = ATS/self.n_cities # normalized to 0~1
        if self.stats:
            self.nATS.append(normalized_ATS)
        return normalized_ATS

    # Adapt ACS parameters: rho, q0
    def adapt_params(self, nATS):
        if self.no_improvement_count > 50:
            if nATS > 0.95:
                if self.p < 0.5:
                    self.p += 0.05
                    self.no_improvement_count = 0
            elif nATS < 0.90:
                if self.p > 0.05:
                    self.p -= 0.05
                    self.no_improvement_count = 0

    # Perform two_opt local search
    def two_opt(self, tour, dist):
        local_best_dist = dist
        local_best_tour = tour

        improved = True
        while improved:
            improved = False
            for i in range(local_best_tour.shape[0]-1):
                c1 = local_best_tour[i]
                c2 = local_best_tour[i+1]
                for c3 in self.neighbour_matrix[c2,:]:
                    if self.dist_matrix[c1, c2] <= self.dist_matrix[c2, c3]:
                        break
                    c3_idx = find_index(local_best_tour, c3)
                    new_tour = two_opt_swap(local_best_tour, i, c3_idx)

                    # Increment number of fitness evaluations. For HAACS, increment only when running
                    # with the count_2opt_fitness flag
                    new_dist = get_tour_dist(self.dist_matrix, new_tour[:-1])
                    if self.alg != 'HAACS' or self.count_2opt_fitness:
                        self.fitness_evals += 1

                    if new_dist < local_best_dist:
                        improved = True
                        local_best_dist = new_dist
                        local_best_tour = new_tour

        if local_best_dist < self.best_dist:
            self.best_dist = local_best_dist
            self.best_tour = local_best_tour

    def stop_condition(self):
        if self.fitness_evals >= self.max_fitness_evals:
            return True

        if self.max_fitness_evals == float('inf') and self.iters >= self.max_iters:
            return True

        return False

    def save_and_reset_best(self):
        self.two_opt_best_dist = self.best_dist
        self.two_opt_best_tour = self.best_tour

        self.best_dist = np.inf
        self.best_tour = None

    def set_best(self):
        if self.two_opt_best_dist < self.best_dist:
            self.best_dist = self.two_opt_best_dist
            self.best_tour = self.two_opt_best_tour


    def solve_ACS(self):
        while not self.stop_condition():
            self.iters += 1
            self.clear_ant_data()

            for i in range(1, self.n_cities):
                self.choose_next_city()
                self.local_update()
            self.return_to_start()
            self.local_update()

            self.find_best_ant(adaptive=False)
            self.global_update()            

            if self.stats:
                self.normalized_average_tour_similarity()

            if self.verbose and self.iters % 10 == 0:
                print('Iteration: {}'.format(self.iters))
                print('Current Best Distance: {}'.format(self.best_dist))
                print()

    def solve_AACS(self):
        while not self.stop_condition():
            self.iters += 1
            self.clear_ant_data()

            for i in range(1, self.n_cities):
                self.choose_next_city()
                self.local_update()
            self.return_to_start()
            self.local_update()
            
            self.find_best_ant()
            self.global_update()

            # Adaptively adjust parameters every 10 iterations
            if self.iters % 10 == 0:
                nATS = self.normalized_average_tour_similarity()
                self.adapt_params(nATS)
            elif self.stats:
                self.normalized_average_tour_similarity()

            if self.verbose and self.iters % 10 == 0:
                print('Iteration: {}'.format(self.iters))
                print('q0: {}, p: {}, nATS: {}'.format(self.q0, self.p, nATS))
                print('Current Best Distance: {}'.format(self.best_dist))
                print()

    def solve_2opt(self):
        while not self.stop_condition():
            self.iters += 1
            tour = get_nn_tour(np.copy(self.dist_matrix), self.n_cities)

            # Increment number of fitness evaluations
            dist = get_tour_dist(self.dist_matrix, tour[:-1])
            self.fitness_evals += 1
            
            self.two_opt(tour, dist)

            if self.verbose and self.iters % 10 == 0:
                print('Iteration: {}'.format(self.iters))
                print('Current Best Distance: {}'.format(self.best_dist))
                print()

    def solve_HAACS(self):
        is_2opt = None
        while not self.stop_condition():
            self.iters += 1

            # 2opt for the initial 'n_2opt_iters' iterations
            if self.iters <= self.n_2opt_iters:
                is_2opt = True
                tour = get_nn_tour(np.copy(self.dist_matrix), self.n_cities)

                # Increment number of fitness evaluations. increment only when running
                # with the count_2opt_fitness flag
                dist = get_tour_dist(self.dist_matrix, tour[:-1])
                if self.count_2opt_fitness:
                    self.fitness_evals += 1
                
                self.two_opt(tour, dist)
                self.global_update()

            # Run AACS for the remaining iterations
            else:
                # save best results from two_opt when switching to AACS
                if is_2opt:
                    self.save_and_reset_best()

                is_2opt = False
                self.clear_ant_data()
                for i in range(1, self.n_cities):
                    self.choose_next_city()
                    self.local_update()
                self.return_to_start()
                self.local_update()

                self.find_best_ant()
                self.global_update()

                # Adaptively adjust parameters every 10 iterations
                if self.iters % 10 == 0:
                    nATS = self.normalized_average_tour_similarity()
                    self.adapt_params(nATS)
                elif self.stats:
                    self.normalized_average_tour_similarity()

            if self.verbose and self.iters % 1 == 0:
                iter_type = '2opt' if is_2opt else 'AACS'
                print('{} Iteration: {}'.format(iter_type, self.iters))
                if is_2opt:
                    print('Current Best Distance: {}'.format(self.best_dist))
                else:
                    print('q0: {}, p: {}, nATS: {}'.format(self.q0, self.p, nATS))
                    print('Current Best Distance: {}'.format(self.best_dist))
                print()
            
            if self.iters % 5 == 0:
                self.print_and_save()
            
        self.set_best()
        

    def solve(self):
        if self.stats:
            self.init_stats()
        if self.verbose:
            print('BEGIN ALGORITHM')
            print('---------------------')

        start_time = time.perf_counter()

        # Run chosen heuristic
        if self.alg == 'ACS':
            self.solve_ACS()
        elif self.alg == 'AACS':
            self.solve_AACS()
        elif self.alg == '2opt':
            self.solve_2opt()
        elif self.alg == 'HAACS':
            self.solve_HAACS()
        else:
            raise Exception('Invalid algorithm name. Options: [ACS | AACS | 2opt | HAACS]')

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        if self.verbose:
            self.print_results(elapsed_seconds)
            sanity_check(self.best_tour[:-1], self.n_cities)
        if self.stats and self.alg in ['ACS', 'AACS', 'HAACS']:
            self.plot_stats()
        if self.graph:
            self.graph_solution()

    
    ### Print/Plot utils ###

    def plot_stats(self):
        plt.plot(self.nATS, label='nATS')
        plt.plot(max_min_norm(self.best_dist_list, 20749, 26000), label='Normalized Best Distance')
        plt.xlabel('# Iterations')
        plt.legend()
        plt.title('nATS and Normalized Best Distance')
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.show()

    def graph_solution(self):
        tour = self.best_tour[:-1]
        g = nx.Graph()
        coords = np.transpose([np.roll(tour, 1), tour])

        for i in range(len(self.coord_list)):
            g.add_node(i, pos=self.coord_list[i])
        for v1, v2 in coords:
            g.add_edge(v1, v2)

        pos = nx.get_node_attributes(g, 'pos')
        nx.draw(g, pos, node_size=5)
        plt.show()

    def print_configs(self, args):
        print('\nCONFIGURATIONS')
        print('---------------------')
        for arg in vars(args):
            print('{}: {}'.format(arg, getattr(args, arg)))

    def print_results(self, elapsed_seconds):
        print('RESULTS')
        print('---------------------')
        print('Seconds Elapsed: {}'.format(elapsed_seconds))
        elapsed_time = time.strftime('%H hours %M minutes %S seconds', time.gmtime(elapsed_seconds))
        print('Time Elapsed: {}'.format(elapsed_time))
        print('Num fitness evals: {}'.format(self.fitness_evals))
        print('Best Tour: {}'.format(self.best_tour[:-1]))
        print('Best Distance: {}'.format(self.best_dist), end='\n\n')

    def print_and_save(self):
        print(self.best_dist)
        np.savetxt('solution.csv', self.best_tour[:-1], fmt='%d')
