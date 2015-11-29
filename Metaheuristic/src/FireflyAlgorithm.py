
from Solution import Solution

import math
import fractions
import numpy
import numpy.random
import scipy.spatial.distance
#import matplotlib.pyplot as plt
import pdb

class FireflyAlgorithm:

    registerEvolution = False
    evolutionLog = {
        'vectors': [],
        'intensity': [],
        'best': [],
        'average': [],
        'distance': [],
        'movedDistance': []
    }

    def __init__(self, dimension, alpha, gamma, beta0=1):
        self._dimension = dimension
        self._alpha = alpha
        self._gamma = fractions.Fraction(gamma)
        self._beta0 = beta0

    @staticmethod
    def logState(fireflies, theBest, movedDistance):
        vectors = [ff.getVectorRep() for ff in fireflies]
        intensityVec = [ff.intensity() for ff in fireflies]
        FireflyAlgorithm.evolutionLog['vectors'].append(vectors)
        FireflyAlgorithm.evolutionLog['intensity'].append(intensityVec)
        FireflyAlgorithm.evolutionLog['best'].append(theBest.intensity())
        FireflyAlgorithm.evolutionLog['average'].append(numpy.mean(intensityVec))
        FireflyAlgorithm.evolutionLog['distance'].append(scipy.spatial.distance.pdist(vectors).sum())
        FireflyAlgorithm.evolutionLog['movedDistance'].append(movedDistance)

    @staticmethod
    def distance(x1,x2):
        # Continuos case:
        #return scipy.spatial.distance.euclidean(x1,x2)
        #return scipy.spatial.distance.sqeuclidean(x1,x2)
        # Discrete case:
        #return ((x1-x2)**2).sum()
        return numpy.abs(x1-x2).sum()

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

        movedDistance = 0
        sortedFireflies = sorted(fireflies, reverse=True)
        theBest = fireflies[0]
        for t in range(maxGeneration):
            # Register the state
            if FireflyAlgorithm.registerEvolution:
                FireflyAlgorithm.logState(sortedFireflies, theBest, movedDistance)
                movedDistance = 0

            for i in range(0, numFireflies):
                changed = False
                for j in range(0, numFireflies):
                    si = fireflies[i]
                    siVector = si.getVectorRep()
                    sj = fireflies[j]
                    if si is not sj:
                        sjVector = sj.getVectorRep()

                        attractivenessInv = self.attractivenessInv(siVector, sjVector)
                        #print("{:d} : {:d}".format(self.distance(siVector, sjVector), attractivenessInv))
                        if (self._beta0 * sj.intensity()) / attractivenessInv > si.intensity():
                            newVector = self.moveTowards(siVector, sjVector, attractivenessInv)

                            # Register total traveled distance
                            if FireflyAlgorithm.registerEvolution:
                                movedDistance += scipy.spatial.distance.pdist(numpy.vstack((siVector, newVector)))[0]
                            fireflies[i] = assignNewFunc(newVector)

                            changed = True
                # If firefly didn't move, move it randomically
                if not changed:
                    siVector = fireflies[i].getVectorRep()
                    newVector = self.moveRandom(siVector)

                    # Register total traveled distance
                    if FireflyAlgorithm.registerEvolution:
                        movedDistance += scipy.spatial.distance.pdist(numpy.vstack((siVector, newVector)))[0]
                    fireflies[i] = assignNewFunc(newVector)

            # Update beta0 vatiable
            if self._beta0 > 1:
                self._beta0 *= 0.99
                if self._beta0 < 1:
                    self._beta0 = 1

            sortedFireflies = sorted(fireflies, reverse=True)
            if sortedFireflies[0].intensity() > theBest.intensity():
                theBest = sortedFireflies[0]
            #print("{:f} => Best: {:f} | ".format(self._beta0, sortedFireflies[0].intensity()))# + str(sortedFireflies[0].getVectorRep()))
        # end optimization loop

        return sortedFireflies, theBest

    def visualize(self, fireflies, r, b):
        pass
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
