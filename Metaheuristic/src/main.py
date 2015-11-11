
from Solution import Solution

R = 100
B = 35

sol = Solution()
sol.randomize(R, B)

print(sol.requestComponent)
print(sol.routesComponent)
print(sol.getRoutes(R))
