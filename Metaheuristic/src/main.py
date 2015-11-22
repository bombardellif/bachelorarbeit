
from Solution import Solution
from FireflyAlgorithm import FireflyAlgorithm
import pdb
import numpy

## __main__
R = 20
B = 5

reqGraph = numpy.random.randint(100, size=(2*R+1)**2).reshape(((2*R+1),(2*R+1)))

fireflyOptimization = FireflyAlgorithm(R+B, 1, 1)

answer = fireflyOptimization.run(99, 20,
    lambda : Solution(reqGraph, R, B).randomize(),
    lambda vector : Solution(reqGraph, R, B, vectorRep=vector)
    )

for a in answer:
    print("Route: ")
    print(a.getRoutes())
    print("Costs: {:d}".format(-a.intensity()))
    print("==================")

'''
sol = Solution(reqGraph, R,B)
sol.randomize()

print(sol.requestComponent)
print(sol.routesComponent)
print(sol.intensity())
'''
