# Mike Carter
# 03.24.2018
# Personal Research
# Evolutionary Optimizer

###############################################################################
#                                                                             #
# Driving home from SF today I was thinking about general artificial intelli- #
# gence, and trying to reconcile thoughts proposed by Superintelligence by    #
# Nick Bostrom with my understanding of how human intelligence evolved. I was #
# thinking vaguely about genetic algorithms, particularly in their applica-   #
# tion to artificial neural networks, and I made a realization. The genetic   #
# algorithms that I have seen so far miss one critical factor: genetic mixing #
# through "sexual" reproduction. This will be a topic of future study, but    #
# I landed on a simpler idea for now. Can I make use of the exploratory power #
# of "sexual reproduction" (its ability to thoroughly and randomly explore    #
# the input space) to develop a new kind of optimization technique that is    #
# highly robust to non-convexity and requires no differentiability in the     #
# cost function?                                                              #
#                                                                             #
# EDIT: Turns out evolutionary optimization is already a pretty developed     #
# topic. This was still a fun experiment, and a cool way of getting some 1st  #
# hand experience with it before I start reading literature!                  #
#                                                                             #
###############################################################################

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Optimization Parameters #
num_candidates_per_generation = 100
num_parents = 10
num_generations =20
p_mutation = .1

# Function to be optimized
N = 100 # dimension of input space
coefficients = np.random.randint(0,10,size=N)


def initialize_candidates() :
    # randomly initialize the candidates to begin "breeding"
    candidates = 2 * np.random.rand(num_candidates_per_generation, N) + 1
    return candidates

def compute_costs(candidates) :
    # compute the cost for each candidate
    cost = np.dot(candidates**2, coefficients)
    return cost

def select_parents(costs) :
    parent_indices = np.argsort(costs)[0:num_parents]
    return parent_indices, parent_indices[0]

def populate_next_generation(parents) :

    # Pass on some randomly selected values from parent 1
    parent1 = np.random.randint(0, num_parents, num_candidates_per_generation)
    reproductive_mask = np.random.choice(a=[True, False], size=(num_candidates_per_generation, N), p=[.5, .5])
    children = reproductive_mask*parents[parent1]

    # Pass on some randomly selected values from parent 2
    parent2 = np.random.randint(0, num_parents, num_candidates_per_generation)
    children += np.logical_not(reproductive_mask)*parents[parent2]

    # Change some values randomly
    mutation_mask = np.random.choice(a=[True, False], size=(num_candidates_per_generation, N), p = [p_mutation, 1-p_mutation])
    children *= np.logical_not(mutation_mask) # clear the values picked to be mutated
    children += mutation_mask*(2*np.random.rand(num_candidates_per_generation,N)-1)

    return children

def evolve_optimal_candidate() :
    # run evolutionary optimization
    candidate_history = np.zeros([num_generations+1, num_candidates_per_generation, N])
    candidate_history[0,:,:] = initialize_candidates()
    best_candidate_history = np.zeros([num_generations+1, N])
    best_cost_history = np.zeros([num_generations+1])
    cost_history = np.zeros([num_generations+1, num_candidates_per_generation])

    for gen in range(num_generations) :
        cost_history[gen,:] = compute_costs(candidate_history[gen,:,:])
        parents, best_parent = select_parents(cost_history[gen,:])
        best_candidate_history[gen] = candidate_history[gen, best_parent, :]
        best_cost_history[gen] = cost_history[gen,best_parent]
        candidate_history[gen+1,:,:] = populate_next_generation(candidate_history[gen,parents,:])
    cost_history[num_generations] = compute_costs(candidate_history[num_generations,:,:])

    return candidate_history, cost_history, best_candidate_history, best_cost_history

def plot_optimization_history(cost_history) :
    plt.plot(range(num_generations+1), cost_history)
    return

# Only works for an input in R2
def plot_optimization_path(candidate_history, cost_history) :
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.linspace(-1, 1, 50)
    Y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(X, Y)
    Z = coefficients[0]*X**2 + coefficients[1]*Y**2

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Plot the path of the optimization for the first candidate in the generation
    xs = candidate_history[:, 0]
    ys = candidate_history[:, 1]
    zs = cost_history
    ax.plot(xs, ys, zs=zs)
    return


if __name__ == '__main__':
    candidate_history, cost_history, best_candidate_history, best_cost_history = evolve_optimal_candidate()
    plot_optimization_history(best_cost_history)
    plot_optimization_path(best_candidate_history, best_cost_history)
    plt.show()
