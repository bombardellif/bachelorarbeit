
from Solution import Solution

import math
import fractions
import numpy
import numpy.random
import scipy.spatial.distance
# import matplotlib.pyplot as plt
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
        'attractMean': [],
        'alpha': []
    }

    def __init__(self, dimension, alpha, alphaDivisor, gamma=1, gammaDenominator=None, beta0=1):
        self._dimension = dimension
        self._alpha = alpha
        self._alphaDivisor = fractions.Fraction(1, alphaDivisor)
        self._alphaStage = 0
        self._currentAlphaDivisor = self._alphaDivisor
        if gammaDenominator is None:
            self._gamma = fractions.Fraction(gamma)
        else:
            self._gamma = fractions.Fraction(1, gammaDenominator)
        self._beta0 = beta0

    @staticmethod
    def logState(fireflies, theBest, movedDistance, changesBecauseIntensity, attractMean, alpha):
        vectors = [ff.getVectorRep().tolist() for ff in fireflies]
        intensityVec = [ff.intensity() for ff in fireflies]
        FireflyAlgorithm.evolutionLog['vectors'].append(vectors)
        FireflyAlgorithm.evolutionLog['intensity'].append(intensityVec)
        FireflyAlgorithm.evolutionLog['best'].append(theBest.intensity())
        FireflyAlgorithm.evolutionLog['average'].append(numpy.mean(intensityVec))
        FireflyAlgorithm.evolutionLog['distance'].append(scipy.spatial.distance.pdist(vectors).sum())
        FireflyAlgorithm.evolutionLog['movedDistance'].append(movedDistance)
        FireflyAlgorithm.evolutionLog['changesBecauseIntensity'].append(changesBecauseIntensity)
        FireflyAlgorithm.evolutionLog['attractMean'].append(attractMean)
        FireflyAlgorithm.evolutionLog['alpha'].append(alpha)

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
        # Single alpha for every component of the vector:
        #return x1 + (diff // attractInv) +\
        #    numpy.rint(((diff % attractInv)/attractInv + self.randomTerm())\
        #    .astype(float)).astype(int)

        # Single alpha for every component of the vector:
        candidate = Solution.createFromRandomVector(x1 + (diff // attractInv),
            self._currentAlphaDivisor,
            numpy.random.normal(size=self._dimension),
            self._alphaStage)

        if not candidate.isInsideDomain() and candidate.canApplyTimeAdjust():
            newRoutesComponent = candidate.satisfyTimeConstraints()
            candidate = Solution(numpy.concatenate((candidate.getVectorRep()[:1], newRoutesComponent)))

        return candidate

    def moveRandom(self, x):
        # Single alpha for every component of the vector:
        #return x + numpy.rint(self.randomTerm()).astype(int)
        # Alpha that varies to each component of the vector:
        candidate = Solution.createFromRandomVector(x,
            self._currentAlphaDivisor,
            numpy.random.normal(size=self._dimension),
            self._alphaStage)

        if not candidate.isInsideDomain() and candidate.canApplyTimeAdjust():
            newRoutesComponent = candidate.satisfyTimeConstraints()
            candidate = Solution(numpy.concatenate((candidate.getVectorRep()[:1], newRoutesComponent)))

        return candidate

    def run(self, maxGeneration, numFireflies):
        fireflies = [Solution().randomize() for i in range(numFireflies)]
        '''
        FireflyAlgorithm.fig = plt.figure()
        plt.ion()
        plt.show()
        '''

        currentAlpha = self._alpha[0]
        alphaDecay = 95

        movedDistance = 0
        changesBecauseIntensity = 0
        attractAcc = 0
        sortedFireflies = sorted(fireflies, reverse=True)
        theBest = fireflies[0]

        for t in range(maxGeneration):
            # Register the state
            if FireflyAlgorithm.registerEvolution:
                FireflyAlgorithm.logState(sortedFireflies, theBest, movedDistance, changesBecauseIntensity, attractAcc, currentAlpha)
                movedDistance = 0
                changesBecauseIntensity = 0
                attractAcc = 0

            for i in range(0, numFireflies):
                if fireflies[i] is not theBest:
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
                                fireflies[i] = self.moveTowards(siVector, sjVector, attractivenessInv)

                                # Register total traveled distance
                                if FireflyAlgorithm.registerEvolution:
                                    movedDistance += scipy.spatial.distance.pdist(numpy.vstack((siVector, fireflies[i].getVectorRep())))[0]
                                    changesBecauseIntensity += 1

                                changed = True
                    # If firefly didn't move, move it randomly
                    if not changed:
                        siVector = fireflies[i].getVectorRep()
                        fireflies[i] = self.moveRandom(siVector)

                        # Register total traveled distance
                        if FireflyAlgorithm.registerEvolution:
                            movedDistance += scipy.spatial.distance.pdist(numpy.vstack((siVector, fireflies[i].getVectorRep())))[0]

            # Update beta0 variable
            if self._beta0 > 1:
                self._beta0 *= 0.99
                if self._beta0 < 1:
                    self._beta0 = 1
            # Update alpha variable
            if currentAlpha > 1:
                currentAlpha *= alphaDecay/100
                # if alpha arrives to 1, change to next stage of alpha cooling
                if currentAlpha < 1:
                    if self._alphaStage < self._alpha.size-1:
                        self._alphaStage += 1
                        currentAlpha = self._alpha[self._alphaStage]
                        alphaDecay = 90
                        self._currentAlphaDivisor = self._alphaDivisor
                    else:
                        currentAlpha = 1
                        self._currentAlphaDivisor = None
                else:
                    self._currentAlphaDivisor = fractions.Fraction(self._currentAlphaDivisor.numerator * alphaDecay,
                                                        self._currentAlphaDivisor.denominator * 100)

            sortedFireflies = sorted(fireflies, reverse=True)
            if sortedFireflies[0].intensity() > theBest.intensity():
                theBest = sortedFireflies[0]
        # end optimization loop
        print("ALPHA: "+str(currentAlpha))
        return sortedFireflies, theBest
