
from Solution import Solution

import math
import fractions
import numpy
import numpy.random
import scipy.spatial.distance
import matplotlib.pyplot as plt
import pdb

class FireflyAlgorithm:

    fig = None
    registerEvolution = False
    evolutionLog = {
        'vectors': [],
        'intensity': [],
        'best': [],
        'average': [],
        'distance': [],
        'movedDistance': [],
        'changesBecauseIntensity': [],
        'attractMean': []
    }

    def __init__(self, dimension, alpha, gamma=1, gammaDenominator=None, beta0=1):
        self._dimension = dimension
        self._alpha = alpha
        if gammaDenominator is None:
            self._gamma = fractions.Fraction(gamma)
        else:
            self._gamma = fractions.Fraction(1, gammaDenominator)
        self._beta0 = beta0

    @staticmethod
    def logState(fireflies, theBest, movedDistance, changesBecauseIntensity, attractMean):
        vectors = [ff.getVectorRep() for ff in fireflies]
        intensityVec = [ff.intensity() for ff in fireflies]
        FireflyAlgorithm.evolutionLog['vectors'].append(vectors)
        FireflyAlgorithm.evolutionLog['intensity'].append(intensityVec)
        FireflyAlgorithm.evolutionLog['best'].append(theBest.intensity())
        FireflyAlgorithm.evolutionLog['average'].append(numpy.mean(intensityVec))
        FireflyAlgorithm.evolutionLog['distance'].append(scipy.spatial.distance.pdist(vectors).sum())
        FireflyAlgorithm.evolutionLog['movedDistance'].append(movedDistance)
        FireflyAlgorithm.evolutionLog['changesBecauseIntensity'].append(changesBecauseIntensity)
        FireflyAlgorithm.evolutionLog['attractMean'].append(attractMean)

        #FireflyAlgorithm.visualize(numpy.array(vectors))

    @staticmethod
    def visualize(vectors):
        plt.clf()
        ax = FireflyAlgorithm.fig.add_subplot(111)
        ax.set_xlim(0, 2e7)
        #ax.set_ylim(0, 3*1e5)
        plt.scatter(vectors[:,0], vectors[:,1])
        FireflyAlgorithm.fig.canvas.draw()

    @staticmethod
    def distance(x1,x2):
        # Continuos case:
        #return scipy.spatial.distance.euclidean(x1,x2)
        #return scipy.spatial.distance.sqeuclidean(x1,x2)
        # Discrete case:
        return ((x1-x2)**2).sum()
        #return numpy.abs(x1-x2).sum()

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
        '''
        FireflyAlgorithm.fig = plt.figure()
        plt.ion()
        plt.show()
        '''

        movedDistance = 0
        changesBecauseIntensity = 0
        attractAcc = 0
        sortedFireflies = sorted(fireflies, reverse=True)
        theBest = fireflies[0]

        for t in range(maxGeneration):
            # Register the state
            if FireflyAlgorithm.registerEvolution:
                FireflyAlgorithm.logState(sortedFireflies, theBest, movedDistance, changesBecauseIntensity, attractAcc)
                movedDistance = 0
                changesBecauseIntensity = 0
                attractAcc = 0

            for i in range(0, numFireflies):
                changed = False
                for j in range(0, numFireflies):
                    si = fireflies[i]
                    siVector = si.getVectorRep()
                    siIntensity = si.intensity()

                    sj = fireflies[j]
                    if si is not sj and sj.isInsideDomain():
                        sjVector = sj.getVectorRep()

                        attractivenessInv = self.attractivenessInv(siVector, sjVector)
                        if FireflyAlgorithm.registerEvolution:
                            attractAcc += attractivenessInv

                        if siIntensity == 0 or (self._beta0 * sj.intensity()) / attractivenessInv > siIntensity:
                            newVector = self.moveTowards(siVector, sjVector, attractivenessInv)

                            # Register total traveled distance
                            if FireflyAlgorithm.registerEvolution:
                                movedDistance += scipy.spatial.distance.pdist(numpy.vstack((siVector, newVector)))[0]
                                changesBecauseIntensity += 1
                            newVector[1:] = 0
                            fireflies[i] = assignNewFunc(newVector)

                            changed = True
                # If firefly didn't move, move it randomically
                if not changed:
                    siVector = fireflies[i].getVectorRep()
                    newVector = self.moveRandom(siVector)
                    newVector[1:] = 0

                    # Register total traveled distance
                    if FireflyAlgorithm.registerEvolution:
                        movedDistance += scipy.spatial.distance.pdist(numpy.vstack((siVector, newVector)))[0]
                    fireflies[i] = assignNewFunc(newVector)

            # Update beta0 variable
            if self._beta0 > 1:
                self._beta0 *= 0.99
                if self._beta0 < 1:
                    self._beta0 = 1
            # Update alpha variable
            if self._alpha > 1:
                self._alpha *= 0.97
                if self._alpha < 1:
                    self._alpha = 1

            sortedFireflies = sorted(fireflies, reverse=True)
            if sortedFireflies[0].intensity() > theBest.intensity():
                theBest = sortedFireflies[0]
        # end optimization loop

        return sortedFireflies, theBest
