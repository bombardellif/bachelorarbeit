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
'../Data/chairedistributique/data/darp/tabu/pr01'
'../Data/chairedistributique/data/darp/tabu/pr02'
'../Data/chairedistributique/data/darp/tabu/pr03'
'../Data/chairedistributique/data/darp/tabu/pr04'
'../Data/chairedistributique/data/darp/tabu/pr05'
'../Data/chairedistributique/data/darp/tabu/pr06'
'../Data/chairedistributique/data/darp/tabu/pr07'
'../Data/chairedistributique/data/darp/tabu/pr08'
'../Data/chairedistributique/data/darp/tabu/pr09'
'../Data/chairedistributique/data/darp/tabu/pr10'
'../Data/chairedistributique/data/darp/tabu/pr11'
'../Data/chairedistributique/data/darp/tabu/pr12'
'../Data/chairedistributique/data/darp/tabu/pr13'
'../Data/chairedistributique/data/darp/tabu/pr14'
'../Data/chairedistributique/data/darp/tabu/pr15'
'../Data/chairedistributique/data/darp/tabu/pr16'
'../Data/chairedistributique/data/darp/tabu/pr17'
'../Data/chairedistributique/data/darp/tabu/pr18'
'../Data/chairedistributique/data/darp/tabu/pr19'
'../Data/chairedistributique/data/darp/tabu/pr20'
)

start=$2
format2_dataset=24
#end=43
end=$3
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
  if [ $i -ge $format2_dataset ]; then
    format=2
  else
    format=1
  fi

  #echo "$PYTHON ../Metaheuristic/src/main.py ${filename[$i]} $format ./$i/solution.json > ./$i/output.txt"
  $PYTHON ../Metaheuristic/src/main.py ${filename[$i]} $format ./$i/solution.json > ./$i/output.txt
done
