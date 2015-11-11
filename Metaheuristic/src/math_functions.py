
import numpy
import numpy.random
import scipy
import scipy.misc
import pdb

def numPossibleCombinationsDeterminedBus(numRequests, requestsOut, numBuses):
    return 1 if requestsOut == 0 \
        else scipy.misc.comb(numRequests, requestsOut, exact=True) if numBuses == 2 \
        else scipy.misc.comb(numRequests, requestsOut, exact=True) \
            * sum([numPossibleCombinationsDeterminedBus(requestsOut, i, numBuses-1) for i in range(requestsOut+1)])
    '''tab = ''.join('\t' for i in range(debug))
    if requestsOut == 0:
        print("{:s}C({:d},{:d})--1".format(tab,numRequests,requestsOut))
    elif numBuses == 2:
        print("{:s}C({:d},{:d})".format(tab,numRequests,requestsOut))
    else:
        l2 = [" F({:d},{:d},{:d})".format(requestsOut, i, numBuses-1) for i in range(requestsOut+1)]
        print("{:s}C({:d},{:d}) ".format(tab,numRequests,requestsOut)+str(l2))
        l = [numPossibleCombinationsDeterminedBus(requestsOut, i, numBuses-1, debug+1) for i in range(requestsOut+1)]

    return 1 if requestsOut == 0 \
        else scipy.misc.comb(numRequests, requestsOut, exact=True) if numBuses == 2 \
        else scipy.misc.comb(numRequests, requestsOut, exact=True) \
            * sum(l)'''

def sizeOfFirstComponent(numRequests, numBuses):
    return numPossibleCombinationsDeterminedBus(numRequests, numRequests, numBuses+1)
#    return [numPossibleCombinationsDeterminedBus(numRequests, i, numBuses) for i in range(numRequests+1)]

def sizeOfSecondComponent(numRequests, numBuses):
    return numBuses * ((numRequests**numRequests) * scipy.misc.factorial(numRequests, exact=True))

def generateBusesRoutesFromConfig(config):
    requestsEachBus = scipy.bincount(config)
    sizePossiblePermutations = (requestsEachBus ** requestsEachBus) * scipy.misc.factorial(requestsEachBus)

    routes = [numpy.random.randint(r) for r in sizePossiblePermutations]

    return routes

def requestsFromTreePath(numRequests, path):
    stops = list(range(numRequests, numRequests*2))

    requests = []
    for i in path:
        requests.append(stops[i])
        if stops[i] - numRequests < 0:
            del stops[i]
        else:
            stops[i] -= numRequests
            stops.sort()

    return requests

def routeFromNumber(numRequests, x):
    # Matrix of boarding at each level
    qtdBoardingChildren = numpy.zeros((numRequests+1, numRequests*2), dtype=int)
    filling = numpy.arange(numRequests, 0, -1)
    idx = numpy.arange(0, numRequests*2, 2)
    for i in range(0, numRequests):
        qtdBoardingChildren[i][idx] = filling[i:]
        idx = idx[:-1]+1

    print(qtdBoardingChildren)

    # Matrix of alighting at each level
    qtdAlightingChildren = numpy.zeros((numRequests+1, numRequests*2), dtype=int)
    filling = numpy.full(numRequests, 1, dtype=int)
    idx = numpy.arange(1, numRequests*2, 2)
    for i in range(1, numRequests+1):
        qtdAlightingChildren[i][idx] = filling[i-1:]
        filling += 1
        idx = idx[:-1]+1

    print(qtdAlightingChildren)

    # Matrix of size of each branch at each level
    newColumn = numpy.append([1], numpy.zeros(numRequests, dtype=int))
    sizeMatrix = newColumn[numpy.newaxis].T
    for i in range(numRequests*2 - 1, -1, -1):
        newColumn = \
            numpy.hstack((0, newColumn[:-1])) * qtdAlightingChildren[:,i] \
            + numpy.hstack(( newColumn[1:], 0)) * qtdBoardingChildren[:,i]
        sizeMatrix = numpy.hstack((newColumn[numpy.newaxis].T, sizeMatrix))

    print(sizeMatrix)

    # Determine the path in the generation tree that X represents
    path = []
    row = 0
    infLimit = 0
    for col in range(0, numRequests*2):
        # Distribution of the range of the children
        dist = []
        if row > 0:
            dist.extend([sizeMatrix[row-1, col+1]] * qtdAlightingChildren[row, col])
        if row < sizeMatrix.shape[0]-1:
            dist.extend([sizeMatrix[row+1, col+1]] * qtdBoardingChildren[row, col])

        maxChild = qtdAlightingChildren[row, col] + qtdBoardingChildren[row, col]

        # Find which is the next stop is regarding the current one
        for nthChild in range(0, maxChild):
            if x - infLimit < dist[nthChild]:
                if nthChild < qtdAlightingChildren[row, col]:
                    row -= 1
                else:
                    row += 1
                break
            else:
                infLimit += dist[nthChild]
        path.append(nthChild)
    print(path)

    # Translate the path in the tree to an identifier of the request
    requestsOrder = requestsFromTreePath(numRequests, path)
    print(requestsOrder)

# __main__
R = 100
B = 35

configRequests = numpy.random.randint(B, size=R)
print(configRequests)
pdb.set_trace()
#[numpy.flatnonzero(x) for x in configRequests == [[0],[1],[3]]] << Req per bus
routes = generateBusesRoutesFromConfig(configRequests)
print(routes)
requestsEachBus = scipy.bincount(configRequests)
print(requestsEachBus)
routeFromNumber(requestsEachBus[0], routes[0])
