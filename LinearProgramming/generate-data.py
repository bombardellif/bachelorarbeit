
from RequestsGraph import RequestsGraph

req = RequestsGraph()
req.loadFromFileORLibrary("../Data/chairedistributique/data/darp/branch-and-cut/a2-16", True)
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/a2-16-times.txt")
