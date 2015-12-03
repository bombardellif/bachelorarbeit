
import numpy
import numpy.random
import scipy
import scipy.misc
import random
import pdb

class Solution:

    worstCost = None

    def __init__(self, requestsGraph, totalRequests, totalBuses, vectorRep=None):
        ### Null initialized values ###
        self._busEachRequest = None
        self._numRequestsEachBus = None
        self._requestsEachBus = None
        self._alightMatrix = {}
        self._boardMatrix = {}
        self._childrenSizeMatrix = {}
        self._routeEachBus = None
        self._intensity = None
        ### Argument initialized values ###
        self._requestsGraph = requestsGraph
        self._totalRequests = totalRequests
        self._totalBuses = totalBuses
        if vectorRep is not None:
            self.assignComponentValues(vectorRep)
        else:
            self._vectorRep = None
            self._requestComponent = None
            self._routesComponent = None

    # Estimation of worst cost possible
    @staticmethod
    def determineWorstCost(costMatrix, totalRequests):
        doublePathSize = (totalRequests*2 + 2) * 2
        Solution.worstCost = -(numpy.partition(-costMatrix, doublePathSize, axis=None)[:doublePathSize].sum() // 2)

    # Operators
    def __lt__(self, b):
        return self.intensity() < b.intensity()

    def __le__(self, b):
        return self.intensity() <= b.intensity()

    def __eq__(self, b):
        return self.intensity() == b.intensity()

    def __ne__(self, b):
        return self.intensity() != b.intensity()

    def __ge__(self, b):
        return self.intensity() >= b.intensity()

    def __gt__(self, b):
        return self.intensity() > b.intensity()

    # Lazy getters
    def getVectorRep(self):
        if self._vectorRep is None:
            # Assertion
            assert self._requestComponent is not None
            assert self._routesComponent is not None

            self._vectorRep = numpy.append(self._requestComponent, self._routesComponent)

        return self._vectorRep

    def getBusEachRequest(self):
        assert self._requestComponent is not None

        # Check cache
        if self._busEachRequest is None:
            self._busEachRequest = numpy.empty(self._totalRequests, dtype=int)

            lastBusNumber = self._totalBuses - 1
            permutationNro = self._requestComponent
            divisor = self._totalBuses ** (self._totalRequests-1)
            nroOddsTillHere = 0
            for i in range(self._totalRequests):
                # if the numbers of odds till this iteration is odd, then invert
                # the sequence, instead of 0,1,2... we'll have 3,2,1,0
                if nroOddsTillHere & 0x1:
                    newValue = lastBusNumber - (permutationNro // divisor)
                else:
                    newValue = permutationNro // divisor

                if newValue & 0x1:
                    nroOddsTillHere += 1

                self._busEachRequest[i] = newValue
                permutationNro = permutationNro % divisor
                divisor //= self._totalBuses

        return self._busEachRequest

    def getNumRequestsEachBus(self):
        # Assertion
        assert self._totalBuses is not None

        if self._numRequestsEachBus is None:
            self._numRequestsEachBus = scipy.bincount(self.getBusEachRequest(), minlength=self._totalBuses)

        return self._numRequestsEachBus

    def getRequestsEachBus(self):

        # Check cache
        if self._requestsEachBus is None:

            # 1. Create a matrix like [[0],[1],[2],...] of size N
            # 2. Compare the elements of busEachRequest to this matrix, the comparison
            # returns an array of size N, with True in the positions where busEachRequest
            # equals the number
            # 3. Apply to each element "flatzero" which returns the indices where we have True
            # 4. In the end we have for each bus in the result the respective requests
            self._requestsEachBus = numpy.array([numpy.flatnonzero(v) \
                for v in self.getBusEachRequest() == \
                    numpy.arange(len(self.getNumRequestsEachBus()), dtype=int)[:,numpy.newaxis] ])

        return self._requestsEachBus

    def getAligthMatrixFor(self, numRequests):
        # Check the cache
        if numRequests not in self._alightMatrix:

            qtdAlightingChildren = numpy.zeros((numRequests+1, numRequests*2), dtype=int)
            filling = numpy.ones(numRequests, dtype=int)
            idx = numpy.arange(1, numRequests*2, 2, dtype=int)
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
            filling = numpy.arange(numRequests, 0, -1, dtype=int)
            idx = numpy.arange(0, numRequests*2, 2, dtype=int)
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

            # Generate matrix of children sizes iteratively
            newColumn = numpy.zeros(numRequests+1, dtype=object)
            newColumn[0] = 1
            sizeMatrix = numpy.empty((numRequests+1, numRequests*2 + 1), dtype=object)
            sizeMatrix[:,-1] = newColumn
            # Auxiliar vectors for calculations
            # shiftDownVector has always 0 in the first cell
            shiftDownVector = numpy.zeros(numRequests+1, dtype=object)
            # shiftUpVector has always 0 in the last cell
            shiftUpVector = numpy.zeros(numRequests+1, dtype=object)
            for i in range(numRequests*2 - 1, -1, -1):
                shiftDownVector[1:] = newColumn[:-1]
                shiftUpVector[:-1] = newColumn[1:]
                newColumn = \
                    shiftDownVector * qtdAlightingChildren[:,i] \
                    + shiftUpVector * qtdBoardingChildren[:,i]
                sizeMatrix[:,i] = newColumn

            # Update cache
            self._childrenSizeMatrix[numRequests] = sizeMatrix

        return self._childrenSizeMatrix[numRequests]

    # Randomize methods
    def _generateRoutesComponent(self):
        sizePermutationsEachBus = self.getSizeDomainEachBus()

        # Small numbers implementation
        '''return numpy.fromiter((numpy.random.randint(size) for size in sizePermutationsEachBus),
            dtype=int, count=len(sizePermutationsEachBus))'''
        # Huge numbers implementation
        return numpy.array([random.randint(0, size-1) for size in sizePermutationsEachBus], dtype=object)

    def randomize(self):
        assert self._totalRequests is not None
        assert self._totalBuses is not None

        # Randomize the distribution of buses to requests
        #self._requestComponent = numpy.random.randint(self._totalBuses, size=self._totalRequests)
        self._requestComponent = numpy.random.randint(self._totalBuses**self._totalRequests, size=1)

        # Randomize valid routes for each bus, given the distribution requests X bus
        self._routesComponent = self._generateRoutesComponent()
        self._routesComponent[:] = 0

        return self

    # Domain transformation methods
    def isInsideDomain(self):
        return (self._requestComponent < self._totalBuses**self._totalRequests).all()\
            and (self._requestComponent >= 0).all()\
            and (self._routesComponent < self.getSizeDomainEachBus()).all()\
            and (self._routesComponent >= 0).all()

    def getSizeDomainEachBus(self):
        return [self.getChildrenSizeMatrixFor(nRequests)[0,0] for nRequests in self.getNumRequestsEachBus()]

    def assignComponentValues(self, newVector):
        # Discretize the input vector
        #newVector = numpy.rint(vector).astype(int)

        # Clip the requests domain [0,No_BUSES)
        # CHANGE: Don't clip anymore, isntead, these will have intensity=0
        '''numpy.clip(newVector[:self._totalRequests],
            0, self._totalBuses-1,
            out=newVector[:self._totalRequests])
        '''
        #self._requestComponent = newVector[:self._totalRequests].astype(int)
        self._requestComponent = newVector[:1]

        # Clip the routes domain [0,Size_Domain_For_Each_Bus)
        # CHANGE: Don't clip anymore, isntead, these will have intensity=0
        '''numpy.clip(newVector[self._totalRequests:],
            0, self.getSizeDomainEachBus(),
            out=newVector[self._totalRequests:])
        '''
        self._routesComponent = newVector[1:]

        # Assign vector attribute
        self._vectorRep = newVector

    @staticmethod
    def _transformTreePathToRelativeRequest(numRequests, path):
        stops = list(range(numRequests, numRequests*2))

        # Translate the path in the tree to an identifier of the request
        # relative to the bus (i.e. sequential, don't care which request it actually got)
        requests = []
        for i in path:
            requests.append(stops[i])
            if stops[i] < numRequests:
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
        assert self._routesComponent is not None
        assert bus < len(numRequestsEachBus)

        # Find out the path in the tree of possibilities
        path = self._transformRouteComponentToTreePath(
            numRequestsEachBus[bus],
            self._routesComponent[bus])

        # Find out the order of picking up and delivering the requests of this bus
        return self._transformTreePathToRelativeRequest(numRequestsEachBus[bus], path)

    def getRoutes(self):
        # Assertion
        assert self._routesComponent is not None
        assert self._totalRequests is not None

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
                numpy.append(requestsEachBus[bus],requestsEachBus[bus]+self._totalRequests
                )[self._transformRouteComponentToRoute(bus)]
                    for bus in range(len(self._routesComponent))
            ])

        return self._routeEachBus

    # Intensity calculation functions
    def intensity(self):
        # Assertion
        assert self._requestsGraph is not None

        # Check cache
        if self._intensity is None:
            if self.isInsideDomain():
                routes = self.getRoutes() + 1

                rows = []
                columns = []
                for r in routes:
                    if r.size != 0:
                        rows += [0] + r.tolist()
                        columns += r.tolist() + [0]

                cost = self._requestsGraph[rows, columns].sum()
                self._intensity = Solution.worstCost - cost
            else:
                self._intensity = 0

        return self._intensity
