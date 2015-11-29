
import numpy
import scipy.spatial.distance
import csv
import pdb

from Request import Request

class RequestsGraph:

    garage = [52.3034705503,10.6835020305]

    def __init__(self):
        self.adjacencyCostMatrix = None
        self.requests = []

    def loadFromFile(self, filename):

        fromLocation = []
        toLocation = []
        with open(filename, newline='') as csvFile:
            csvFile.readline()
            spam = csv.reader(csvFile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_NONNUMERIC)
            for row in spam:
                fromLocation.append([row[3],row[4]])
                toLocation.append([row[1],row[2]])
                self.requests.append(Request((row[3],row[4]), (row[1],row[2])))

        # Calculate the distance between any pair of locations
        locations = numpy.concatenate(([RequestsGraph.garage], toLocation, fromLocation))
        self.adjacencyCostMatrix = scipy.spatial.distance.cdist(locations, locations)
