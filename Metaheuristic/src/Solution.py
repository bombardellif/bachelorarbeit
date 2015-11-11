
import numpy
import numpy.random
import scipy
import scipy.misc
import pdb

class Solution:

    def __init__(self, requestComponent=None, routesComponent=None):
        self.requestComponent = requestComponent
        self.routesComponent = routesComponent
        self._numRequestsEachBus = None
        self._requestsEachBus = None
        self._alightMatrix = {}
        self._boardMatrix = {}
        self._childrenSizeMatrix = {}
        self._routeEachBus = None
        self._cost = None

    # Lazy getters
    def getNumRequestsEachBus(self):
        # Assertion
        assert self.requestComponent is not None

        if self._numRequestsEachBus is None:
            self._numRequestsEachBus = scipy.bincount(self.requestComponent)

        return self._numRequestsEachBus

    def getRequestsEachBus(self):
        # Assertion
        assert self.requestComponent is not None

        # Check cache
        if self._requestsEachBus is None:

            # 1. Create a matrix like [[0],[1],[2],...] of size N
            # 2. Compare the elements of equestComponent to this matrix, the comparison
            # returns an array of size N, with True in the positions where requestComponent
            # equals the number
            # 3. Apply to each element "flatzero" which returns the indices where we have True
            # 4. In the end we have for each bus in the result the respective requests
            self._requestsEachBus = numpy.array([numpy.flatnonzero(v) \
                for v in self.requestComponent == \
                    numpy.arange(len(self.getNumRequestsEachBus()))[:,numpy.newaxis] ])

        return self._requestsEachBus

    def getAligthMatrixFor(self, numRequests):
        # Check the cache
        if numRequests not in self._alightMatrix:

            qtdAlightingChildren = numpy.zeros((numRequests+1, numRequests*2), dtype=int)
            filling = numpy.full(numRequests, 1, dtype=int)
            idx = numpy.arange(1, numRequests*2, 2)
            for i in range(1, numRequests+1):
                qtdAlightingChildren[i][idx] = filling[i-1:]
                filling += 1
                idx = idx[:-1]+1

            # Update cache
            self._alightMatrix[numRequests] = qtdAlightingChildren

        return self._alightMatrix[numRequests]

    def getBoardMatrixFor(self, numRequests):
        # Check the cache
        if numRequests not in self._boardMatrix:

            qtdBoardingChildren = numpy.zeros((numRequests+1, numRequests*2), dtype=int)
            filling = numpy.arange(numRequests, 0, -1)
            idx = numpy.arange(0, numRequests*2, 2)
            for i in range(0, numRequests):
                qtdBoardingChildren[i][idx] = filling[i:]
                idx = idx[:-1]+1

            # Update cache
            self._boardMatrix[numRequests] = qtdBoardingChildren

        return self._boardMatrix[numRequests]

    def getChildrenSizeMatrixFor(self, numRequests):
        # Check the cache
        if numRequests not in self._childrenSizeMatrix:

            # Get auxiliar matrices
            qtdAlightingChildren = self.getAligthMatrixFor(numRequests)
            qtdBoardingChildren = self.getBoardMatrixFor(numRequests)

            # generate matrix of children sizes iteratively
            newColumn = numpy.append([1], numpy.zeros(numRequests, dtype=int))
            sizeMatrix = newColumn[numpy.newaxis].T
            for i in range(numRequests*2 - 1, -1, -1):
                newColumn = \
                    numpy.hstack((0, newColumn[:-1])) * qtdAlightingChildren[:,i] \
                    + numpy.hstack(( newColumn[1:], 0)) * qtdBoardingChildren[:,i]
                sizeMatrix = numpy.hstack((newColumn[numpy.newaxis].T, sizeMatrix))

            # Update cache
            self._childrenSizeMatrix[numRequests] = sizeMatrix

        return self._childrenSizeMatrix[numRequests]

    # Randomize methods
    def _generateRoutesComponent(self):
        sizePermutationsEachBus = [self.getChildrenSizeMatrixFor(nRequests)[0,0]\
            for nRequests in self.getNumRequestsEachBus()]

        return numpy.fromiter((numpy.random.randint(size) for size in sizePermutationsEachBus),
            dtype=int, count=len(sizePermutationsEachBus))

    def randomize(self, totalRequests, totalBuses):
        # Randomize the distribution of buses t requests
        self.requestComponent = numpy.random.randint(totalBuses, size=totalRequests)

        # Randomize valid routes for each bus, given the distribution requests X bus
        self.routesComponent = self._generateRoutesComponent()

    # Domain transformation methods
    @staticmethod
    def _transformTreePathToRelativeRequest(numRequests, path):
        stops = list(range(numRequests, numRequests*2))

        # Translate the path in the tree to an identifier of the request
        # relative to the bus (i.e. sequential, don't care which request it actually got)
        requests = []
        for i in path:
            requests.append(stops[i])
            if stops[i] - numRequests < 0:
                del stops[i]
            else:
                stops[i] -= numRequests
                stops.sort()

        return requests

    def _transformRouteComponentToTreePath(self, numRequests, value):
        # Get auxiliar matrices
        qtdAlightingChildren = self.getAligthMatrixFor(numRequests)
        qtdBoardingChildren = self.getBoardMatrixFor(numRequests)
        childrenSizeMatrix = self.getChildrenSizeMatrixFor(numRequests)

        # Determine the path in the generation tree that "value" represents
        path = []
        row = 0
        infLimit = 0
        for col in range(0, numRequests*2):
            # Distribution of the range of the children
            dist = []
            if row > 0:
                dist.extend([childrenSizeMatrix[row-1, col+1]] * qtdAlightingChildren[row, col])
            if row < childrenSizeMatrix.shape[0]-1:
                dist.extend([childrenSizeMatrix[row+1, col+1]] * qtdBoardingChildren[row, col])

            maxChild = qtdAlightingChildren[row, col] + qtdBoardingChildren[row, col]

            # Find which is the next stop is regarding the current one
            for nthChild in range(0, maxChild):
                if value - infLimit < dist[nthChild]:
                    if nthChild < qtdAlightingChildren[row, col]:
                        row -= 1
                    else:
                        row += 1
                    break
                else:
                    infLimit += dist[nthChild]
            path.append(nthChild)

        return path

    def _transformRouteComponentToRoute(self, bus):
        # Local variable
        numRequestsEachBus = self.getNumRequestsEachBus()

        # Assertion
        assert self.routesComponent is not None
        assert bus < len(numRequestsEachBus)

        # Find out the path in the tree of possibilities
        path = self._transformRouteComponentToTreePath(
            numRequestsEachBus[bus],
            self.routesComponent[bus])

        # Find out the order of picking up and delivering the requests of this bus
        return self._transformTreePathToRelativeRequest(numRequestsEachBus[bus], path)

    def getRoutes(self, totalRequests):
        # Assertion
        assert self.routesComponent is not None

        # Check cache
        if self._routeEachBus is None:
            requestsEachBus = self.getRequestsEachBus()

            # 1. Consider that the id of each request represents the "get out" point
            # and then, the id + totalNumberOfRequests represents the "get in" point
            # 2. We get the ordered ids of the requests of each bus and extend this
            # array adding in the tail the number codes for the "get in" points
            # 3. The function of transformation will return an array with the same
            # logic, but with sequencial numbers starting from 0, we just have to use it
            # as index for the just constructed array, then we have the route of the bus
            self._routeEachBus = numpy.array([
                numpy.append(requestsEachBus[bus],requestsEachBus[bus]+totalRequests
                )[self._transformRouteComponentToRoute(bus)]
                    for bus in range(len(self.routesComponent))
            ])

        return self._routeEachBus

    # Cost calculation functions
    def getCost(self):
        if self._cost is None:
            self._cost = 0

        return self._cost
