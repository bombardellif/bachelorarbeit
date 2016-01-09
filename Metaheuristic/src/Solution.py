
import numpy
import numpy.random
import scipy
import scipy.misc
import random
import fractions
import pdb

class Solution:

    # Arguments of the solutions
    requestGraph = None
    totalRequests = None
    totalBuses = None
    maxCapacity = None

    # Cache matrices
    alightMatrix = {}
    boardMatrix = {}
    childrenSizeMatrix = {}

    # Auxiliar values
    randomPrecisionMult = 10 ** 9
    sizeDomainRequestComponent = None
    worstCost = None

    def __init__(self, vector=None):
        ### Solution related values ###
        self._busEachRequest = None
        self._numRequestsEachBus = None
        self._requestsEachBus = None
        self._routeEachBus = None
        self._intensity = None
        ### Search space related values ###
        if vector is None:
            self._vectorRep = None
            self._requestComponent = None
            self._routesComponent = None
        else:
            self.assignComponentValues(vector)

    # Estimation of worst cost possible
    @staticmethod
    def determineWorstCost():
        doublePathSize = (Solution.totalRequests*2 + 2) * 2
        Solution.worstCost = -(numpy.partition(-Solution.requestGraph.costMatrix, doublePathSize, axis=None)[:doublePathSize].sum() // 2)

    @staticmethod
    def initializeClass():
        assert Solution.totalRequests is not None
        assert Solution.totalBuses is not None

        Solution.sizeDomainRequestComponent = Solution.totalBuses ** Solution.totalRequests

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
            self._busEachRequest = numpy.empty(Solution.totalRequests, dtype=int)

            lastBusNumber = Solution.totalBuses - 1
            permutationNro = self._requestComponent
            divisor = Solution.totalBuses ** (Solution.totalRequests-1)
            nroOddsTillHere = 0
            for i in range(Solution.totalRequests):
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
                divisor //= Solution.totalBuses

        return self._busEachRequest

    def getNumRequestsEachBus(self):
        # Assertion
        assert Solution.totalBuses is not None

        if self._numRequestsEachBus is None:
            self._numRequestsEachBus = scipy.bincount(self.getBusEachRequest(), minlength=Solution.totalBuses)

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

    @staticmethod
    def getAligthMatrixFor(numRequests):
        # Check the cache
        if numRequests not in Solution.alightMatrix:

            qtdAlightingChildren = numpy.zeros((numRequests+1, numRequests*2), dtype=int)
            filling = numpy.ones(numRequests, dtype=int)
            idx = numpy.arange(1, numRequests*2, 2, dtype=int)

            for i in range(1, min(numRequests+1, Solution.maxCapacity)):
                qtdAlightingChildren[i][idx] = filling[i-1:]
                filling += 1
                idx = idx[:-1]+1

            # Update cache
            Solution.alightMatrix[numRequests] = qtdAlightingChildren

        return Solution.alightMatrix[numRequests]

    @staticmethod
    def getBoardMatrixFor(numRequests):
        # Check the cache
        if numRequests not in Solution.boardMatrix:

            qtdBoardingChildren = numpy.zeros((numRequests+1, numRequests*2), dtype=int)
            filling = numpy.arange(numRequests, 0, -1, dtype=int)
            idx = numpy.arange(0, numRequests*2, 2, dtype=int)

            for i in range(0, min(numRequests, Solution.maxCapacity)):
                qtdBoardingChildren[i][idx] = filling[i:]
                idx = idx[:-1]+1

            # Update cache
            Solution.boardMatrix[numRequests] = qtdBoardingChildren

        return Solution.boardMatrix[numRequests]

    @staticmethod
    def getChildrenSizeMatrixFor(numRequests):
        # Check the cache
        if numRequests not in Solution.childrenSizeMatrix:

            # Get auxiliar matrices
            qtdAlightingChildren = Solution.getAligthMatrixFor(numRequests)
            qtdBoardingChildren = Solution.getBoardMatrixFor(numRequests)

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
            Solution.childrenSizeMatrix[numRequests] = sizeMatrix

        return Solution.childrenSizeMatrix[numRequests]

    # Randomize methods
    @staticmethod
    def createFromRandomVector(vector, alphaDivisor, randomVector, alphaStage):
        if alphaDivisor is None:
            new = Solution(vector + numpy.rint(randomVector).astype(int))
        else:
            # Calculate new requests component
            newVector = numpy.zeros(vector.shape, dtype=object)
            if alphaStage == 0:
                newVector[0] = vector[0] \
                    + (Solution.sizeDomainRequestComponent * (alphaDivisor.numerator * int(round(randomVector[0] * Solution.randomPrecisionMult)))) \
                        // (alphaDivisor.denominator * Solution.randomPrecisionMult)
            else:
                newVector[0] = vector[0] + int(round(randomVector[0]))

            candidate = Solution(newVector)

            # Set the rest of the vector if the first component is valid
            if candidate.isInsideDomain():
                if alphaStage == 0:
                    #newVector[1:] = vector[1:] + numpy.rint(randomVector[1:]).astype(int)
                    newVector[1:] = 0
                else:
                    newVector[1:] = vector[1:] \
                        + (candidate.getSizeDomainEachBus() * (alphaDivisor.numerator * numpy.rint(randomVector[1:] * Solution.randomPrecisionMult).astype(int).astype(object))) \
                            // (alphaDivisor.denominator * Solution.randomPrecisionMult)
            else:
                newVector[1:] = vector[1:]
            new = Solution(newVector)

        return new

    def _generateRoutesComponent(self):
        sizePermutationsEachBus = self.getSizeDomainEachBus()

        # Small numbers implementation
        '''return numpy.fromiter((numpy.random.randint(size) for size in sizePermutationsEachBus),
            dtype=int, count=len(sizePermutationsEachBus))'''
        # Huge numbers implementation
        return numpy.array([random.randint(0, size-1) for size in sizePermutationsEachBus], dtype=object)

    def randomize(self):
        assert Solution.totalRequests is not None
        assert Solution.totalBuses is not None

        # Randomize the distribution of buses to requests
        #self._requestComponent = numpy.random.randint(Solution.totalBuses, size=Solution.totalRequests)
        self._requestComponent = numpy.array([random.randint(0, Solution.sizeDomainRequestComponent-1)], dtype=object)

        # Randomize valid routes for each bus, given the distribution requests X bus
        self._routesComponent = self._generateRoutesComponent()

        return self

    # Domain transformation methods
    def isInsideDomain(self):
        return (self._requestComponent < Solution.sizeDomainRequestComponent).all()\
            and (self._requestComponent >= 0).all()\
            and (self._routesComponent < self.getSizeDomainEachBus()).all()\
            and (self._routesComponent >= 0).all()\
            and self.matchCapacityContraint()#\
            #and self.matchTimeConstraints()

    def getSizeDomainEachBus(self):
        return [self.getChildrenSizeMatrixFor(nRequests)[0,0] for nRequests in self.getNumRequestsEachBus()]

    def matchCapacityContraint(self):
        routes = self.getRoutes()

        match = True
        for route in routes:
            if route.size:
                # Where passangers get in we count 1 to the load and when get out we count -1
                if (numpy.where(route >= Solution.totalRequests, 1, -1)
                .cumsum() > Solution.maxCapacity).any():
                    match = False
                    break
        return match

    def matchTimeConstraints(self):
        routes = self.getRoutes()
        rows, columns = self.getRoutesInEdges(concatenated=False)

        match = True
        for i in range(len(rows)):
            minPossible = Solution.requestGraph.timeMatrix[[0] + rows[i], [0] + columns[i]].cumsum()
            limits = Solution.requestGraph.timeConstraints[rows[i] + [0]]

            # Minimum feasible time regarding to the "min" constraint of time
            # here we add in the total travel waiting times to match this constraint
            waiting = 0
            for j in range(minPossible.size):
                currentMinTime = minPossible[j] + waiting
                vertexMinTime = limits[j, 0]
                if vertexMinTime > currentMinTime:
                    waiting += vertexMinTime - currentMinTime
                    minPossible[j] = vertexMinTime
                else:
                    minPossible[j] = currentMinTime

            # The minimum possible track must match the "max" constraint of time
            if (minPossible > limits[:, 1]).any():
                match = False
                break

            # The maximum travel time of the car has a constraint too
            if minPossible[-1] > Solution.requestGraph.totalTime:
                match = False
                break

        return match

    def assignComponentValues(self, newVector):
        # Clip the requests domain [0,Size_Domain_Request_Permutation)
        numpy.clip(newVector[:1],
            -1, Solution.sizeDomainRequestComponent,
            out=newVector[:1])
        self._requestComponent = newVector[:1]

        if newVector[0] >= 0 and newVector[0] < Solution.sizeDomainRequestComponent:
            # Clip the routes domain [0,Size_Domain_For_Each_Bus)
            # CHANGE: Don't clip anymore, instead, these will have intensity=0
            numpy.clip(newVector[1:],
                -1, self.getSizeDomainEachBus(),
                out=newVector[1:])

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
            # If it's a alighting, delete this request because it's now delivered
            # else, set this to the number correspondent to the future alighting
            # important: sort the list of pendent request, so that the exits stay first
            if stops[i] < numRequests:
                del stops[i]
            else:
                stops[i] -= numRequests
                stops.sort()

        return requests

    def _transformRouteComponentToTreePath(self, numRequests, value):
        # Get auxiliar matrices
        qtdAlightingChildren = Solution.getAligthMatrixFor(numRequests)
        qtdBoardingChildren = Solution.getBoardMatrixFor(numRequests)
        childrenSizeMatrix = Solution.getChildrenSizeMatrixFor(numRequests)

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
        assert Solution.totalRequests is not None

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
                numpy.append(requestsEachBus[bus],requestsEachBus[bus]+Solution.totalRequests
                )[self._transformRouteComponentToRoute(bus)]
                    for bus in range(len(self._routesComponent))
            ])

        return self._routeEachBus

    def getRoutesInEdges(self, concatenated=True):
        routes = self.getRoutes() + 1

        rows = []
        columns = []
        for r in routes:
            if r.size != 0:
                if concatenated:
                    rows += [0] + r.tolist()
                    columns += r.tolist() + [0]
                else:
                    rows.append([0] + r.tolist())
                    columns.append(r.tolist() + [0])

        return rows, columns

    # Intensity calculation functions
    def intensity(self):
        # Assertion
        assert Solution.requestGraph is not None

        # Check cache
        if self._intensity is None:
            if self.isInsideDomain():
                rows, columns = self.getRoutesInEdges()

                cost = Solution.requestGraph.costMatrix[rows, columns].sum()
                self._intensity = Solution.worstCost - cost
            else:
                self._intensity = 0

        return self._intensity
