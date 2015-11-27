
from Solution import Solution

import math
import fractions
import numpy
import numpy.random
import scipy.spatial.distance
import matplotlib.pyplot as plt
import pdb

class FireflyAlgorithm:

    def __init__(self, dimension, alpha, gamma):
        self._dimension = dimension
        self._alpha = alpha
        self._gamma = fractions.Fraction(gamma)
        self.teste = []

    @staticmethod
    def distance(x1,x2):
        # Continuos case:
        #return scipy.spatial.distance.euclidean(x1,x2)
        #return scipy.spatial.distance.sqeuclidean(x1,x2)
        # Discrete case:
        return ((x1-x2)**2).sum()

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

    def moveRandom(self, x):
        return x + numpy.rint(self.randomTerm()).astype(int)

    def run(self, maxGeneration, numFireflies, generateFunc, assignNewFunc):
        fireflies = [generateFunc() for i in range(numFireflies)]

        '''self.fig = plt.figure()
        plt.ion()
        plt.show()'''

        for t in range(maxGeneration):
            for i in range(0, numFireflies):
                changed = False
                si = fireflies[i]
                siVector = si.getVectorRep()
                for j in range(0, numFireflies):
                    sj = fireflies[j]
                    if si is not sj:
                        sjVector = sj.getVectorRep()

                        attractivenessInv = self.attractivenessInv(siVector, sjVector)
                        if sj.intensity() / attractivenessInv > si.intensity():
                            newVector = self.moveTowards(siVector, sjVector, attractivenessInv)
                            fireflies[i] = assignNewFunc(newVector)

                            changed = True
                # If firefly didn't move, move it randomically
                if not changed:
                    newVector = self.moveRandom(siVector)
                    fireflies[i] = assignNewFunc(newVector)

            test_evolution = sorted(fireflies, reverse=True)
            self.visualize(test_evolution, 15, 3)
            print("Best: {:f} | ".format((test_evolution[0].intensity())) + str(test_evolution[0].getVectorRep()))
        # end optimization loop

        return sorted(fireflies, reverse=True)

    def visualize(self, fireflies, r, b):
        '''lis = []
        for ff in fireflies:
            vec = ff.getVectorRep()
            x = numpy.vander([2], r).dot(numpy.array(vec[:r]).T)[0]
            y = numpy.vander([2], b).dot(numpy.array(vec[r:]).T)[0]
            lis.append([x,y])

        plt.clf()
        ax = self.fig.add_subplot(111)
        ax.set_xlim(0, 4*1e4)
        ax.set_ylim(0, 3*1e5)
        plt.plot(*zip(*lis), marker='o', linestyle='None')
        self.fig.canvas.draw()
        '''
        ffs = numpy.empty((20,r+b), dtype=object)
        for i in range(len(fireflies)):
            ffs[i,:] = fireflies[i].getVectorRep()

        print("Total Distance: {:f}".format(scipy.spatial.distance.pdist(ffs).sum()))
