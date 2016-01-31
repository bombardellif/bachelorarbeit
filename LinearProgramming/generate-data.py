
from RequestsGraph import RequestsGraph

req = RequestsGraph()
req.loadFromFileORLibrary("../Data/chairedistributique/data/darp/branch-and-cut/a2-16", False)
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/a2-16-times.txt")

req.loadFromFileORLibrary("../Data/test-1", False)
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/test-1-times.txt")

req.loadFromFileORLibrary("../Data/test-2", False)
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/test-2-times.txt")

req.loadFromFileORLibrary("../Data/test-3", False)
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/test-3-times.txt")

req.loadFromFileORLibrary("../Data/test-4", False)
#req.loadFromFile("../Data/datasets-2015-11-06-aot-tuberlin/15-2015-11-06.csv")
req.writeToFile("./v01/test-4-times.txt")
