# bachelorarbeit
Files for my bachelor thesis at the TU Berlin.

### Needed Tools for the Metaheuristic Implementation
- Python 3.3+
- NumPy 1.10+
- SciPy 0.16+
- Matplotlib 1.15+

### Needed Tools for the Linear Programming
- GNU Linear Programming Kit (GLPK)

### Basic Setup for the Metaheuristic
Setup of the envirnoment in a Redhat-like linux distribution.
```
# yum install epel-release
# yum install gcc-c++ gcc-gfortran lapack-devel atlas-devel
# yum install python34 python34-setuptools python34-devel
# easy_install-3.4 pip
# pip3.4 install numpy
# pip3.4 install scipy
```
### Basic Setup for the Visualization
```
# yum install libpng-devel freetype-devel pygtk
# pip3.4 install matplotlib
```

### Metaheuristic Solver Usage
```
$ python3.4 Metaheuristic/src/main.py -h
usage: main.py [-h] [-v] data {1,2} output
Dial-a-Ride Solver.
positional arguments:
  data           Instance file path
  {1,2}          Format of the data file. Either 1 or 2.
  output         Output file path
optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose  Show graph windows with the evolution.
```
### Solutions Visualization Usage
```
$ python3.4 graphs.py -h
usage: graphs.py [-h] data {1,2} solution
Dial-a-Ride Graphs Generator.
positional arguments:
  data        Instance file path
  {1,2}       Format of the data file. Either 1 or 2.
  solution    Solution file path
optional arguments:
  -h, --help  show this help message and exit
```
