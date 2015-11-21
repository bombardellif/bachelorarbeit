
from Solution import Solution
from FireflyAlgorithm import FireflyAlgorithm
import pdb
import numpy

## __main__
R = 5
B = 2

reqGraph = numpy.random.randint(100, size=11**2).reshape((11,11))

fireflyOptimization = FireflyAlgorithm(R+B, 1, 1)

answer = fireflyOptimization.run(10, 10,
    lambda : Solution(reqGraph, R, B).randomize(),
    lambda vector : Solution(reqGraph, R, B, vectorRep=vector)
    )

for a in answer:
    print(a.intensity())
'''
sol = Solution(reqGraph, R,B)
sol.randomize()

print(sol.requestComponent)
print(sol.routesComponent)
print(sol.intensity())
'''
