
import pdb
import math
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
req.loadFromFile("../../Data/datasets-2015-11-06-aot-tuberlin/15.2-2015-11-06.csv")

# Attributes
Solution.requestGraph = req.adjacencyCostMatrix
Solution.totalRequests = R
Solution.totalBuses = B
Solution.initializeClass()

# Estimate superior limit for the Utility function
Solution.determineWorstCost()
# Define Parameters
#alpha = numpy.zeros(1+B, dtype=int)
alpha = numpy.array([(B**R) // 10, 6679735362203904000000 // 10], dtype=object)
#alpha = 681080400 // 10
alphaDivisor = 10
#gamma = 1/math.sqrt(B**R)
gammaDenominator = int(round(((B**R)*6679735362203904000000) ** (1/2)))
beta0 = 1
maxGeneration = 1000
numFireflies = 40

# Instanciate Solver
FireflyAlgorithm.registerEvolution = True
fireflyOptimization = FireflyAlgorithm(1+B, alpha, alphaDivisor, gammaDenominator=gammaDenominator, beta0=beta0)

# Run solver
solutions,theBest = fireflyOptimization.run(maxGeneration, numFireflies)

# Show Results
'''
plt.ioff()
plt.show()
'''

for sol in solutions:
    if sol.isInsideDomain():
        print("Route: ")
        print(sol.getRoutes())
    print("Vector: ")
    print(sol.getVectorRep())
    print("Intensity: {:f}".format(sol.intensity()))
    print("==================")

print("THE BEST: ")
print("Route: ")
print(theBest.getRoutes())
print("Vector: ")
print(theBest.getVectorRep())
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

    plt.figure()
    data = FireflyAlgorithm.evolutionLog['changesBecauseIntensity']
    plt.xlabel('Iteration')
    plt.ylabel('Changes Because of Intensity')
    plt.plot(data)

    plt.figure()
    data = FireflyAlgorithm.evolutionLog['attractMean']
    plt.xlabel('Iteration')
    plt.ylabel('Mean of the Attraction')
    plt.plot(data)

    plt.figure()
    data = FireflyAlgorithm.evolutionLog['alpha']
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
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

    '''
    plt.figure().gca(projection='3d')
    data = numpy.array([sol.getVectorRep() for sol in solutions]).astype(float) / 1e15
    plt.scatter(data[:,0], data[:,1], data[:,2])
    '''

    plt.show()

'''
plt.ioff()
plt.show()'''
