
from Solution import Solution

import math
import numpy
import numpy.random
import scipy.spatial.distance
import pdb

class FireflyAlgorithm:

    def __init__(self, dimension, alpha, gamma, beta0=1):
        self._dimension = dimension
        self._alpha = alpha
        self._gamma = gamma
        self._beta0 = beta0

    @staticmethod
    def distance(x1,x2):
        return scipy.spatial.distance.euclidean(x1,x2)

    def attractiveness(self, x1, x2):
        return self._beta0 * math.exp((-self._gamma) * (self.distance(x1,x2)**2))

    def randomTerm(self):
        return numpy.random.normal(size=self._dimension)

    def moveTowards(self, x1, x2):
        return x1 + self.attractiveness(x1,x2) * (x2-x1) + self._alpha * self.randomTerm()

    def run(self, maxGeneration, numFireflies, generateFunc, assignNewFunc):
        fireflies = [generateFunc() for i in range(numFireflies)]

        for t in range(maxGeneration):
            for i in range(0, numFireflies):
                for j in range(0,numFireflies):
                    xi = fireflies[i]
                    xj = fireflies[j]
                    if xi is not xj:
                        if xj.intensity() * self.attractiveness(xi.getVectorRep(), xj.getVectorRep()) > xi.intensity():
                            newVector = self.moveTowards(xi.getVectorRep(), xj.getVectorRep())
                            fireflies[i] = assignNewFunc(newVector)
        # end optimization loop

        return sorted(fireflies)
