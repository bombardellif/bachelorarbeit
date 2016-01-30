#!/bin/sh

filename=(
'../Data/chairedistributique/data/darp/branch-and-cut/a2-16'
'../Data/chairedistributique/data/darp/branch-and-cut/a2-20'
'../Data/chairedistributique/data/darp/branch-and-cut/a2-24'
'../Data/chairedistributique/data/darp/branch-and-cut/a3-18'
'../Data/chairedistributique/data/darp/branch-and-cut/a3-24'
'../Data/chairedistributique/data/darp/branch-and-cut/a3-30'
'../Data/chairedistributique/data/darp/branch-and-cut/a3-36'
'../Data/chairedistributique/data/darp/branch-and-cut/a4-16'
'../Data/chairedistributique/data/darp/branch-and-cut/a4-24'
'../Data/chairedistributique/data/darp/branch-and-cut/a4-32'
'../Data/chairedistributique/data/darp/branch-and-cut/a4-40'
'../Data/chairedistributique/data/darp/branch-and-cut/a4-48'
'../Data/chairedistributique/data/darp/branch-and-cut/a5-40'
'../Data/chairedistributique/data/darp/branch-and-cut/a5-50'
'../Data/chairedistributique/data/darp/branch-and-cut/a5-60'
'../Data/chairedistributique/data/darp/branch-and-cut/a6-48'
'../Data/chairedistributique/data/darp/branch-and-cut/a6-60'
'../Data/chairedistributique/data/darp/branch-and-cut/a6-72'
'../Data/chairedistributique/data/darp/branch-and-cut/a7-56'
'../Data/chairedistributique/data/darp/branch-and-cut/a7-70'
'../Data/chairedistributique/data/darp/branch-and-cut/a7-84'
'../Data/chairedistributique/data/darp/branch-and-cut/a8-64'
'../Data/chairedistributique/data/darp/branch-and-cut/a8-80'
'../Data/chairedistributique/data/darp/branch-and-cut/a8-96'
)

start=0
end=23
PYTHON=$1

for i in $(eval echo "{$start..$end}");
do
  echo $i
  # Create result directory if not existent yet
  if [ ! -d "./$i" ]; then
    #echo "mkdir $i"
    mkdir $i
  fi
  # Run the instance
  echo "$PYTHON ../Metaheuristic/src/main.py ${filename[$i]} 1 ./$i/solution.json > ./$i/output.txt"
  #python3 ../Metaheuristic/src/main.py ${filename[$i]} 1 ./$i/solution.json > ./$i/output.txt
done
