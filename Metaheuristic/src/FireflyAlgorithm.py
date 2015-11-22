
from Solution import Solution

import math
import fractions
import numpy
import numpy.random
import scipy.spatial.distance
import pdb

class FireflyAlgorithm:

    def __init__(self, dimension, alpha, gamma):
        self._dimension = dimension
        self._alpha = alpha
        self._gamma = fractions.Fraction(gamma)

    @staticmethod
    def distance(x1,x2):
        # Continuos case:
        #return scipy.spatial.distance.euclidean(x1,x2)
        # Discrete case:
        return scipy.spatial.distance.sqeuclidean(x1,x2)

    def attractivenessInv(self, x1, x2):
        # Continuous case:
        #return self._beta0 * math.exp((-self._gamma) * (self.distance(x1,x2)**2))
        # Discrete case:
        return 1 + (self._gamma.numerator * self.distance(x1,x2)) // self._gamma.denominator

    def randomTerm(self):
        return self._alpha * numpy.random.normal(size=self._dimension)

    def moveTowards(self, x1, x2, attractInv):
        # Continuous case:
        #return x1 + self.attractiveness(x1,x2) * (x2-x1) + self.randomTerm()
        # Discrete case:
        diff = x2-x1
        return x1 + (diff // attractInv) +\
            numpy.rint(((diff % attractInv)/attractInv + self.randomTerm())\
            .astype(float)).astype(int)

    def run(self, maxGeneration, numFireflies, generateFunc, assignNewFunc):
        fireflies = [generateFunc() for i in range(numFireflies)]

        for t in range(maxGeneration):
            for i in range(0, numFireflies):
                for j in range(0, numFireflies):
                    si = fireflies[i]
                    sj = fireflies[j]
                    if si is not sj:
                        attractivenessInv = self.attractivenessInv(si.getVectorRep(), sj.getVectorRep())
                        if sj.intensity() // attractivenessInv > si.intensity():
                            newVector = self.moveTowards(si.getVectorRep(), sj.getVectorRep(), attractivenessInv)
                            fireflies[i] = assignNewFunc(newVector)
        # end optimization loop

        return sorted(fireflies, reverse=True)
