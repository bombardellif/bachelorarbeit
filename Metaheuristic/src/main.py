
import pdb
import numpy
import matplotlib.pyplot as plt

from Solution import Solution
from FireflyAlgorithm import FireflyAlgorithm
from RequestsGraph import RequestsGraph

## __main__
R = 15
B = 3

# Load Requests File
req = RequestsGraph()
req.loadFromFile("../../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")

# Estimate superior limit for the Utility function
reqGraph = req.adjacencyCostMatrix
Solution.determineWorstCost(reqGraph, R)

# Define Parameters
alpha = numpy.zeros(R+B, dtype=int)
alpha[R:] = 1
gamma = 1e-11
beta0 = 1
maxGeneration = 500
numFireflies = 40

# Instanciate Solver
FireflyAlgorithm.registerEvolution = True
fireflyOptimization = FireflyAlgorithm(R+B, alpha, gamma, beta0)

# Run solver
solutions,theBest = fireflyOptimization.run(maxGeneration, numFireflies,
    lambda : Solution(reqGraph, R, B).randomize(),
    lambda vector : Solution(reqGraph, R, B, vectorRep=vector)
    )

# Show Results
for sol in solutions:
    print("Route: ")
    print(sol.getRoutes())
    print("Intensity: {:f}".format(sol.intensity()))
    print("==================")

print("THE BEST: ")
print("Route: ")
print(theBest.getRoutes())
print("Intensity: {:f}".format(theBest.intensity()))
print("==================")

# Visualize Evolution
if FireflyAlgorithm.registerEvolution:
    plt.figure()
    data = FireflyAlgorithm.evolutionLog['best']
    plt.xlabel('Iteration')
    plt.ylabel('The all best')
    plt.plot(data)

    plt.figure()
    data = FireflyAlgorithm.evolutionLog['average']
    plt.xlabel('Iteration')
    plt.ylabel('Average')
    plt.plot(data)

    plt.figure()
    data = [intes[0] for intes in FireflyAlgorithm.evolutionLog['intensity']]
    plt.xlabel('Iteration')
    plt.ylabel('Iteration best')
    plt.plot(data)

    plt.figure()
    data = FireflyAlgorithm.evolutionLog['distance']
    plt.xlabel('Iteration')
    plt.ylabel('Total distance')
    plt.plot(data)

    plt.figure()
    data = FireflyAlgorithm.evolutionLog['movedDistance']
    plt.xlabel('Iteration')
    plt.ylabel('Moved distance')
    plt.plot(data)

    style = ['-og','-ob','-oy']
    i = 0
    for route in theBest.getRoutes():
        plt.figure()
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')

        xdata = numpy.empty(len(route)+2)
        ydata = numpy.empty(len(route)+2)
        xdata[0] = RequestsGraph.garage[0]
        ydata[0] = RequestsGraph.garage[1]

        alightIdx = []
        j = 1
        for stop in route:
            if (stop >= R):
                requestNumber = stop - R
                xdata[j] = req.requests[requestNumber].fromLocation[0]
                ydata[j] = req.requests[requestNumber].fromLocation[1]
            else:
                requestNumber = stop
                xdata[j] = req.requests[requestNumber].toLocation[0]
                ydata[j] = req.requests[requestNumber].toLocation[1]
                alightIdx.append(j)
            j += 1
        xdata[j] = RequestsGraph.garage[0]
        ydata[j] = RequestsGraph.garage[1]

        plt.plot(xdata, ydata, style[i], xdata[0], ydata[0], 'ok', xdata[alightIdx], ydata[alightIdx], 'or')
        i += 1

    plt.show()

'''
plt.ioff()
plt.show()'''
