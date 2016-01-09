##### Program

# Global params
param N integer;										# Number of requests
param L integer;										# Maximum ride time of requests
param M integer;										# Large integer number

# Sets
set P := 1..N;
set D := N+1..2*N;
set V := {0, 2*N+1} union P union D;
set EDGE within {i in V, j in V: i != j};
set BUS;

# Params from the graph
#param Cost_edge {EDGE} >= 0;
param Time_edge {EDGE} >=0;

# Params from the requests
param Lim_min {V} >= 0;					# Time intervals
param Lim_max {V} >= 0;
param Q {V};										# Load of the request
param Dur {V} >= 0;							# Service duration

# Params from the buses
param Cap {BUS} > 0;						# Capacity of the bus
param Tim {BUS} >= 0;						# Maximum Ride time of a bus

# Variables
var x {EDGE, BUS} binary;				# Routes of a bus
var u {V, BUS} >= 0;						# Time that a bus serves at a vertex
var w {V, BUS} >= 0;						# Load of a bus when leaving a vertex
var r {P, BUS} >= 0;						# ride time of a request

# Objective Function
minimize costs:
	(sum {b in BUS, (i,j) in EDGE} Time_edge[i,j] * x[i,j,b]);

# Constraints
		## Each request is served by some vehicle
subject to S1 {i in P}:
	sum {b in BUS, j in V diff {i}} x[i,j,b] = 1;

		## Vehicle starts and ends at the depot
subject to S2_1 {b in BUS}:
	sum {i in V diff {0}} x[0,i,b] = 1;
subject to S2_2 {b in BUS}:
	sum{i in V diff {2*N+1}} x[i,2*N+1,b] = 1;

		## Each request is served by only one vehicle and it is alway the same
subject to S3 {i in P, b in BUS}:
	sum {j in V diff {i}} x[i,j,b] - sum{j in V diff {N+i}} x[N+i,j,b] = 0;
subject to S4 {i in P union D, b in BUS}:
	sum {j in V diff {i}} x[j,i,b] - sum{j in V diff {i}} x[i,j,b] = 0;

		## Define start of service, vehicle load and ride time
subject to S5_1 {(i,j) in EDGE, b in BUS}:
	u[j,b] >= u[i,b] + Dur[i] + Time_edge[i,j] - M*(1 - x[i,j,b]);
#subject to S5_2 {(i,j) in EDGE, b in BUS}:
#	U[i,j,b] >= max(0, Lim_max[i] + Dur[i] + Time_edge[i,j] - Lim_min[i]);

subject to S6_1 {(i,j) in EDGE, b in BUS}:
	w[j,b] >= w[i,b] + Q[j] - M*(1 - x[i,j,b]);
#subject to S6_2 {(i,j) in EDGE, b in BUS}:
#	W[i,j,b] >= min(Cap[b], Cap[b] + Q[i]);

subject to S7 {i in P, b in BUS}:
	r[i,b] >= u[N+i,b] - (u[i,b] + Dur[i]);

		## Ensure feasibility
subject to S8 {b in BUS}:
	u[2*N+1,b] - u[0,b] <= Tim[b];
subject to S9 {i in V, b in BUS}:
	Lim_min[i] <= u[i,b] <= Lim_max[i];
subject to S10 {i in P, b in BUS}:
	Time_edge[i,N+i] <= r[i,b] <= L;
subject to S11 {i in V, b in BUS}:
	max(0, Q[i]) <= w[i,b] <= min(Cap[b], Cap[b] + Q[i]);
