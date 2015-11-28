
import pdb
import numpy
import matplotlib.pyplot as plt

from Solution import Solution
from FireflyAlgorithm import FireflyAlgorithm
from RequestsGraph import RequestsGraph

## __main__
R = 15
B = 3

req = RequestsGraph()
req.loadFromFile("../../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")

reqGraph = req.adjacencyCostMatrix

alpha = numpy.zeros(R+B, dtype=int)
alpha[R:] = 1
fireflyOptimization = FireflyAlgorithm(R+B, alpha, 1e-8, 2)

answer,theBest = fireflyOptimization.run(500, 20,
    lambda : Solution(reqGraph, R, B).randomize(),
    lambda vector : Solution(reqGraph, R, B, vectorRep=vector)
    )

for a in answer:
    print("Route: ")
    print(a.getRoutes())
    print("Intensity: {:f}".format(a.intensity()))
    print("==================")
print("THE BEST: ")
print("Route: ")
print(theBest.getRoutes())
print("Intensity: {:f}".format(theBest.intensity()))
print("==================")
'''
plt.ioff()
plt.show()'''
'''
sol = Solution(reqGraph, R,B)
sol.randomize()

print(sol.requestComponent)
print(sol.routesComponent)
print(sol.intensity())
'''
