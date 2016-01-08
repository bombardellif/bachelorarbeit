
from RequestsGraph import RequestsGraph

req = RequestsGraph()
req.loadFromFileORLibrary("../Data/pr01-reduced-mini")
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/test-data-03-times.txt")
