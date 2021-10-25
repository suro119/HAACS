import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import networkx as nx

def parse_commandline_args():
    parser = argparse.ArgumentParser()
    # TSP file
    parser.add_argument('tsp_file', type=str, help='input tsp file')
    # ACO Parameters
    parser.add_argument('-q', '--exploit_tendency', type=float, default=0.9, help='exploitation tendency. range: [0,1]')
    parser.add_argument('--beta', type=float, default=2, help='importance of distance in choosing the next city. range: all reals')
    parser.add_argument('--rho', type=float, default=0.1, help='portion of pheromone decrease in local update. range: [0,1]')
    parser.add_argument('--alpha', type=float, default=0.1, help='portion of pheromone replacement in global update. range: [0,1]')
    # General Parameters
    parser.add_argument('-a', '--algorithm', type=str, default='HAACS', help='decide on the algorithm [ACS | AACS | 2opt | HAACS]')
    parser.add_argument('-p', '--population', type=int, default=10, help='population (number of ants)')
    parser.add_argument('-f', '--max_fitness_evals', type=int, default=float('inf'), help='number of fitness evaluations. If set, takes precedence over max_iters')
    parser.add_argument('-i', '--max_iters', type=int, default=1000, help='number of iterations for ACS, for debugging purposes')
    # HAACS-only Parameters
    parser.add_argument('-t', '--n_2opt_iters', type=int, default=50, help='number of 2opt iterations before converting to AACS')
    parser.add_argument('-c', '--count_2opt_fitness', default=False, action='store_true', help='does not count fitness evaluations used in 2opt (only for HAACS)')
    # Test Parameters
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='print helpful information to stdout')
    parser.add_argument('-s', '--get_stats', default=False, action='store_true', help='get nATS and best_distance statistics and plot the results')
    parser.add_argument('-g', '--graph', default=False, action='store_true', help='graph best solution')
    args = parser.parse_args()
    return args

def read_tsp(tsp_file):
    # Read tsp file and append (x,y) tuples into a python list
    tsp_coordinate_list = []
    with open(tsp_file) as f:
        reader = csv.reader(f, delimiter=' ')
        for _ in range(6):
            next(reader)
        for row in reader:
            if row[0] == 'EOF':
                break
            tsp_coordinate_list.append(tuple(row[1:3]))
    
    tsp_coordinate_list = np.asarray(tsp_coordinate_list).astype(np.float)
    return tsp_coordinate_list

def get_nn_tour(dist_matrix, n_cities):
    tour = []
    current_city = np.random.choice(n_cities)
    tour.append(current_city)
    for _ in range(n_cities-1):
        next_city = np.argmin(dist_matrix[current_city,:])
        tour.append(next_city)
        dist_matrix[current_city, :] = np.inf
        dist_matrix[:, current_city] = np.inf
        current_city = next_city
    tour.append(tour[0])
    tour = np.array(tour, dtype=int) # not specifying dtype=int will potentially cause an error in HAACS
    return tour

def get_edge_set(tour):
    edge_list = np.append(np.transpose([tour]), np.transpose([np.roll(tour, 1)]), axis=1)
    flipped_edge_list = np.flip(edge_list, axis=1)
    edge_list = np.append(edge_list, flipped_edge_list, axis=0)
    return edge_list

def get_common_edge_count(set1, set2):
    set1_view = set1.view([('',set1.dtype)]*set1.shape[1]).reshape(-1,1)
    set2_view = set2.view([('',set2.dtype)]*set2.shape[1]).reshape(-1,1)
    intersection = np.intersect1d(set1_view, set2_view)

    assert not intersection.shape[0] % 2, 'intersection should always return even number of edges due to symmetry'

    common_edge_count = intersection.shape[0]//2
    return common_edge_count

# Caution: initial city should not be repeated at the end!
def get_tour_dist(dist_matrix, tour):
    return sum(dist_matrix[np.roll(tour, 1), tour])

def find_index(arr, val):
    for i in range(arr.shape[0]):
        if arr[i] == val:
            return i

def two_opt_swap(tour, i, j):
    new_tour = np.copy(tour)
    if i > j:
        temp = i
        i = j
        j = temp
    new_tour[i+1:j+1] = np.flip(new_tour[i+1:j+1])
    return new_tour

# Performs max_min normalization (for plotting best distance with respect to normalized average tour similarity)
def max_min_norm(arr, min_val, max_val):
    arr = np.array(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def sanity_check(tour, n_cities):
    sorted_tour = np.sort(tour)
    all_cities = np.arange(n_cities)
    sane = np.array_equal(sorted_tour, all_cities)
    if not sane:
        print('tour: {}'.format(tour))
        print('sorted tour: {}'.format(sorted_tour))
        print('len tour: {}'.format(tour.shape[0]))
        print('num_cities: {}'.format(n_cities))
        raise Exception('not all cities included and\or duplicate cities')

    
