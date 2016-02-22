
import pdb
import numpy
import argparse
import json
import matplotlib.pyplot as plt

from Solution import Solution
from RequestsGraph import RequestsGraph

parser = argparse.ArgumentParser(description='Dial-a-Ride Graphs Generator.')
parser.add_argument('data', help="Instance file path")
parser.add_argument('format', help='Format of the data file. Either 1 or 2.', type=int, choices=[1,2])
parser.add_argument('solution', help="Solution file path")

args = parser.parse_args()

#Initialize
req = RequestsGraph()
successLoadData = req.loadFromFileORLibrary(args.data, firstDestiny=True, dataset=args.format)
if not successLoadData:
    print('Error loading data. Exiting...')
    sys.exit(0)

R = req.numOfRequests
B = req.numOfVehicles
Solution.requestGraph = req
Solution.totalRequests = R
Solution.totalBuses = B
Solution.maxCapacity = req.capacity
Solution.initializeClass()

# Estimate superior limit for the Utility function
Solution.determineWorstCost()

# Load JSON data
evolutionLog = None
with open(args.solution, 'r') as solutionFile:
    evolutionLog = json.load(solutionFile)

# Visualize Evolution
if evolutionLog is not None:
    plt.figure()
    data = evolutionLog['log']['best']
    plt.xlabel('Iteration')
    plt.ylabel('The all best')
    plt.plot(data)

    plt.figure()
    data = evolutionLog['log']['average']
    plt.xlabel('Iteration')
    plt.ylabel('Average')
    plt.plot(data)

    plt.figure()
    data = [intes[0] for intes in evolutionLog['log']['intensity']]
    plt.xlabel('Iteration')
    plt.ylabel('Iteration best')
    plt.plot(data)

    plt.figure()
    data = evolutionLog['log']['distance']
    plt.xlabel('Iteration')
    plt.ylabel('Total distance')
    plt.plot(data)

    plt.figure()
    data = evolutionLog['log']['movedDistance']
    plt.xlabel('Iteration')
    plt.ylabel('Moved distance')
    plt.plot(data)

    plt.figure()
    data = evolutionLog['log']['changesBecauseIntensity']
    plt.xlabel('Iteration')
    plt.ylabel('Changes Because of Intensity')
    plt.plot(data)

    plt.figure()
    data = evolutionLog['log']['attractMean']
    plt.xlabel('Iteration')
    plt.ylabel('Mean of the Attraction')
    plt.plot(data)

    plt.figure()
    data = evolutionLog['log']['alpha']
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.plot(data)

    style = ['-og','-ob','-oy']
    i = 0
    theBest = Solution(numpy.array(evolutionLog['best']))
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
        if i < 2:
            i += 1

    '''
    plt.figure().gca(projection='3d')
    data = numpy.array([sol.getVectorRep() for sol in solutions]).astype(float) / 1e15
    plt.scatter(data[:,0], data[:,1], data[:,2])
    '''

    plt.show()
