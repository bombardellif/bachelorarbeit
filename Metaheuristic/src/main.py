
import pdb
import numpy

from Solution import Solution
from FireflyAlgorithm import FireflyAlgorithm
from RequestsGraph import RequestsGraph

## __main__
R = 100
B = 20

req = RequestsGraph()
req.loadFromFile("../../Data/datasets-2015-11-06-aot-tuberlin/100-2015-11-06.csv")

reqGraph = req.adjacencyCostMatrix

fireflyOptimization = FireflyAlgorithm(R+B, 1, 1)

answer = fireflyOptimization.run(300, 20,
    lambda : Solution(reqGraph, R, B).randomize(),
    lambda vector : Solution(reqGraph, R, B, vectorRep=vector)
    )

for a in answer:
    print("Route: ")
    print(a.getRoutes())
    print("Costs: {:f}".format(-a.intensity()))
    print("==================")

'''
sol = Solution(reqGraph, R,B)
sol.randomize()

print(sol.requestComponent)
print(sol.routesComponent)
print(sol.intensity())
'''
