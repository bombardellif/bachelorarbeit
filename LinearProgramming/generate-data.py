
from RequestsGraph import RequestsGraph

req = RequestsGraph()
req.loadFromFileORLibrary("../Data/chairedistributique/data/darp/branch-and-cut/a2-16")
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/test-data-01-times.txt")
