
import pdb
import math
import numpy
import argparse
import json
import datetime

from Solution import Solution
from FireflyAlgorithm import FireflyAlgorithm
from RequestsGraph import RequestsGraph

## __main__

parser = argparse.ArgumentParser(description='Dial-a-Ride Solver.')
parser.add_argument('data', help="Instance's file path")
parser.add_argument('format', help='Format of the data file. Either 1 or 2.', type=int, choices=[1,2])
parser.add_argument('output', help='Output file path')
parser.add_argument('-v', '--verbose', help="Show graph windows with the evolution.", action="store_true")

args = parser.parse_args()

# Load Requests File
req = RequestsGraph()
#req.loadFromFile("../../Data/datasets-2015-11-06-aot-tuberlin/15.2-2015-11-06.csv")
#req.loadFromFileORLibrary("../../Data/pr01-reduced", firstDestiny=True)
#req.loadFromFileORLibrary("../../Data/chairedistributique/data/darp/tabu/pr01", firstDestiny=True)
req.loadFromFileORLibrary(args.data, firstDestiny=True, dataset=args.format)

# Attributes
R = req.numOfRequests
B = req.numOfVehicles
Solution.requestGraph = req
Solution.totalRequests = R
Solution.totalBuses = B
Solution.maxCapacity = req.capacity
Solution.initializeClass()

# Estimate superior limit for the Utility function
Solution.determineWorstCost()

# out = 0
# suc = 0
# for i in range(10):
#     sol = Solution().randomize()
#     # pdb.set_trace()
#     if not sol.isInsideDomain():
#         out += 1
#         newRoutesComponent = sol.satisfyTimeConstraints()
#         sol = Solution(numpy.concatenate((sol.getVectorRep()[:1], newRoutesComponent)))
#         if sol.isInsideDomain():
#             suc += 1
# print(out)
# print(suc)
# pdb.set_trace()

# Define Parameters
#alpha = numpy.zeros(1+B, dtype=int)
sizeRoutesComponents = Solution.getChildrenSizeMatrixFor(R//B)[0,0] ** B
alpha = numpy.array([(B**R) // 10, sizeRoutesComponents // 10], dtype=object)
#alpha = numpy.array([(B**R), sizeRoutesComponents], dtype=object)
alphaDivisor = 10
#gamma = 1/math.sqrt(B**R)
gammaDenominator = int(round(((B**R) * sizeRoutesComponents) ** (1/2)))
beta0 = 1
maxGeneration = 10
numFireflies = 40

# Instanciate Solver
FireflyAlgorithm.registerEvolution = True
fireflyOptimization = FireflyAlgorithm(1+B, alpha, alphaDivisor, gammaDenominator=gammaDenominator, beta0=beta0)

# Output Parameters
print('============= Parameters ===============')
print('Data file:\t', args.data)
print('Format:\t\t', args.format)
print('Requests:\t', req.numOfRequests)
print('Vehicles:\t', req.numOfVehicles)
print('Capacity:\t', req.capacity)
print('Worst cost estimation:\t', Solution.worstCost)
print('Generations:\t', maxGeneration)
print('Fireflies:\t', numFireflies)

print('============= Running ... ==============')
print(datetime.datetime.now().time().isoformat())

# Run solver
solutions,theBest = fireflyOptimization.run(maxGeneration, numFireflies)

print('============= Finished! ================')
print(datetime.datetime.now().time().isoformat())

# Output Results
print('============= Output ... ===============')
with open(args.output, 'w') as outfile:
    json.dump({
        'solutions': [sol.getVectorRep().tolist() for sol in solutions],
        'best': theBest.getVectorRep().tolist(),
        'log': FireflyAlgorithm.evolutionLog if FireflyAlgorithm.registerEvolution else None
    }, outfile)
    print('File saved in: ', args.output)

print('============= Summary =================')
if FireflyAlgorithm.registerEvolution:
    print('Best initial solution:\t', FireflyAlgorithm.evolutionLog['best'][0])
    print('Best initial cost:\t', Solution.worstCost - FireflyAlgorithm.evolutionLog['best'][0])
print('Best solution:\t', theBest.intensity())
print('Best cost:\t', Solution.worstCost - theBest.intensity())

# Show Results
if args.verbose:
    import matplotlib.pyplot as plt
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
        print("Intensity: {:f} === Cost: {:f}".format(sol.intensity(), Solution.worstCost - sol.intensity()))
        print("==================")

    print("THE BEST: ")
    print("Route: ")
    print(theBest.getRoutes())
    print("Vector: ")
    print(theBest.getVectorRep())
    print("Intensity: {:f} === Cost: {:f}".format(theBest.intensity(), Solution.worstCost - theBest.intensity()))
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
