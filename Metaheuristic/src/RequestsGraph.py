
import numpy
import scipy.spatial.distance
import csv
import re
import pdb

#from Request import Request

class RequestsGraph:

    garage = [52.3034705503,10.6835020305]
    totalTime = 999999

    def __init__(self):
        self.costMatrix = None
        self.timeMatrix = None
        self.timeConstraints = None
        #self.requests = []

    def loadFromFile(self, filename):
        fromLocation = []
        toLocation = []
        with open(filename, newline='') as csvFile:
            csvFile.readline()
            spam = csv.reader(csvFile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_NONNUMERIC)
            for row in spam:
                fromLocation.append([row[3],row[4]])
                toLocation.append([row[1],row[2]])
                #self.requests.append(Request((row[3],row[4]), (row[1],row[2])))

        # Calculate the distance between any pair of locations
        locations = numpy.concatenate(([RequestsGraph.garage], toLocation, fromLocation))
        self.costMatrix = scipy.spatial.distance.cdist(locations, locations)

        # Time between locations
        self.timeMatrix = self.costMatrix

        # Create the array of time constraints
        #self.timeConstraints = numpy.concatenate((endIntervals, startIntervals))

    def loadFromFileORLibrary(self, filename):
        parser = re.compile('\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+-?\d+\s+(\d+)\s+(\d+)')

        isFirst = True
        error = False
        with open(filename, 'r') as textFile:
            for line in textFile:
                if isFirst:
                    isFirst = False

                    firstLine = re.match('(\d+)\s+(\d+)', line)
                    if firstLine:
                        (vehicles,requests) = firstLine.groups()

                        #numRequests = int(requests)*2+1
                        numRequests = int(requests) + 1
                        locations = numpy.empty((numRequests, 2), dtype=float)
                    else:
                        error = True
                        break
                else:
                    parsedLine = parser.match(line)
                    if parsedLine:
                        data = parsedLine.groups()
                        idx = int(data[0])
                        if idx < numRequests:
                            location = [float(data[1]), float(data[2])]
                            locations[idx,:] = location

                            if idx == 0:
                                RequestsGraph.garage = location
                    else:
                        error = True
                        break

        if not error:
            self.costMatrix = scipy.spatial.distance.cdist(locations, locations)

            # Time between locations
            self.timeMatrix = self.costMatrix

    def writeToFile(self, filename):
        with open(filename, 'w') as file:
            for index, x in numpy.ndenumerate(self.timeMatrix):
                file.write("{:d} {:d}\t{:.16f}\n".format(index[0], index[1], x))

            # write the edges to the vertex 2*n+1 required in the LP
            lastVertex = self.timeMatrix.shape[1]
            for index, x in numpy.ndenumerate(self.timeMatrix[:,0]):
                file.write("{:d} {:d}\t{:.16f}\n{:d} {:d}\t{:.16f}\n".format(
                            index[0], lastVertex, x, lastVertex, index[0], x))

            # loop vertex in this last one
            file.write("{:d} {:d}\t{:.16f}".format(lastVertex, lastVertex, 0.0))
