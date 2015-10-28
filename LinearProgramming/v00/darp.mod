##### Program
# Sets
set VERTEX;
set EDGE within {VERTEX,VERTEX};
set REQ;
set BUS;

# param of the Moments
param T > 0 integer;

# Params from the graph
param Cost_edge {EDGE} >= 0;
#param Time_edge {EDGE} >=0;
param Cost_vertex {VERTEX} >= 0;
#param Time_vertex {VERTEX} >= 0;

# Params from the requests
param V_in {REQ,VERTEX} binary;			# Boarding point for the request
param V_out {REQ,VERTEX} binary;		# Alighting point for the request
#param Lim_in {REQ} > 0;
#param Lim_out {REQ} > 0;

# Params from the buses
param G {BUS,VERTEX} binary;				# Garage of the bus
param C {BUS} > 0;									# Capacity of the bus
#param T {BUS} >= 0;									# Boarding time p/ person
#param F {BUS} > 0;									# Fix cost of the bus
#param BLim_in {BUS} > 0;						# Start time of the bus operation
#param BLim_out {BUS} > 0;						# End time of the bus operation

# Variables
var c {BUS,0..T} >= 0;							# Load of bus in the moment t
var r {BUS,0..T,REQ} binary;				# Which bus take each request at each moment
var h {BUS,0..T,VERTEX} binary;			# In which stop is each bus at each moment
var p {BUS,0..T,EDGE} binary;				# In which vertex is each bus at each moment
var aux_bus_in_edge {BUS,0..T,EDGE} binary;
var aux_bus_stopped_vertex {BUS,0..T,VERTEX} binary;
var aux_bus_arrived_vertex {BUS,0..T,VERTEX} binary;
var aux_boarding {BUS,1..T,REQ} binary;
var aux_alighting {BUS,1..T,REQ} binary;
var aux_pickup_consistent {BUS,1..T,REQ,VERTEX} binary;
var aux_leave_consistent {BUS,1..T,REQ,VERTEX} binary;

# Objective Function
minimize costs:
	(sum {b in BUS, t in 0..T, (i,j) in EDGE} Cost_edge[i,j] * p[b,t,i,j])
	+ (sum {b in BUS, t in 0..T, i in VERTEX} Cost_vertex[i] * h[b,t,i]);

# Constraints
		## Initial and final states of the buses
subject to S1 {b in BUS}:
	sum {i in VERTEX} h[b,0,i] * G[b,i] = 1;
subject to S2 {b in BUS}:
	sum {i in VERTEX} h[b,T,i] * G[b,i] = 1;

		## Capacity constraints
subject to S3 {b in BUS, t in 0..T}:
	c[b,t] = sum {i in REQ} r[b,t,i];
subject to S4 {b in BUS, t in 0..T}:
	c[b,t] <= C[b];
subject to S5:
	sum {b in BUS} c[b,0] = 0;

		## Consistency of the result path (p and h)
subject to S6 {b in BUS, t in 0..T}:
	(sum {(i,j) in EDGE} p[b,t,i,j]) + (sum {i in VERTEX} h[b,t,i]) = 1;
subject to S7 {b in BUS, t in 0..T-1}:
	(sum {(i,j) in EDGE} p[b,t,i,j]) + (sum {(i,j) in EDGE} p[b,t+1,i,j]) <= 1;
#subject to S8 {b in BUS, t in 1..T-1}:
#	(sum {(i,j) in EDGE} h[b,t-1,i] * p[b,t,i,j] * h[b,t+1,i])
#		+ (sum {i in VERTEX} h[b,t-1,i] * h[b,t,i]) = 1;
subject to S8_1 {b in BUS, t in 1..T-1, (i,j) in EDGE}:
	aux_bus_in_edge[b,t,i,j] <= (h[b,t-1,i] + p[b,t,i,j] + h[b,t+1,j]) / 3;
subject to S8_2 {b in BUS, t in 1..T-1, (i,j) in EDGE}:
	aux_bus_in_edge[b,t,i,j] >= h[b,t-1,i] + p[b,t,i,j] + h[b,t+1,j] - 2;
subject to S8_3 {b in BUS, t in 1..T, i in VERTEX}:
	aux_bus_stopped_vertex[b,t,i] <= (h[b,t-1,i] + h[b,t,i]) / 2;
subject to S8_4 {b in BUS, t in 1..T, i in VERTEX}:
	aux_bus_stopped_vertex[b,t,i] >= h[b,t-1,i] + h[b,t,i] - 1;
subject to S8_5 {b in BUS, t in 1..T, i in VERTEX}:
	aux_bus_arrived_vertex[b,t,i] <= ((sum {(k,j) in EDGE: j = i} p[b,t-1,k,i]) + h[b,t,i]) / 2;
subject to S8_6 {b in BUS, t in 1..T, i in VERTEX}:
	aux_bus_arrived_vertex[b,t,i] >= (sum {(k,j) in EDGE: j = i} p[b,t-1,k,i]) + h[b,t,i] - 1;
subject to S8_7 {b in BUS, t in 1..T-1}:
	(sum {(i,j) in EDGE} aux_bus_in_edge[b,t,i,j])
		+ (sum {i in VERTEX} aux_bus_stopped_vertex[b,t,i])
		+ (sum {i in VERTEX} aux_bus_arrived_vertex[b,t,i]) = 1;
subject to S8_8 {b in BUS}:
	(sum {i in VERTEX} aux_bus_stopped_vertex[b,T,i])
	+ (sum {i in VERTEX} aux_bus_arrived_vertex[b,T,i]) = 1;

		## Consistency of attending the requests
subject to S9 {t in 0..T, i in REQ}:
	sum {b in BUS} r[b,t,i] <= 1;
subject to S10 {b in BUS, t in 0..T-1, i in REQ}:
	r[b,t,i] + (sum {b_ in BUS diff {b}} r[b_,t+1,i]) -1 <= 0;
#subject to S11 {i in REQ}:
#	(sum {b in BUS, t in 0..T-1} (1-r[b,t,i]) * r[b,t+1,i])
#		* (sum {b in BUS, t in 0..T-1} r[b,t,i] * (1-r[b,t+1,i])) = 1;

subject to S11_1 {i in REQ, b in BUS, t in 1..T}:
	aux_boarding[b,t,i] <= ((1-r[b,t-1,i]) + r[b,t,i]) / 2;
subject to S11_2 {i in REQ, b in BUS, t in 1..T}:
	aux_boarding[b,t,i] >= (1-r[b,t-1,i]) + r[b,t,i] - 1;
subject to S11_3 {i in REQ, b in BUS, t in 1..T}:
	aux_alighting[b,t,i] <= (r[b,t-1,i] + (1-r[b,t,i])) / 2;
subject to S11_4 {i in REQ, b in BUS, t in 1..T}:
	aux_alighting[b,t,i] >= r[b,t-1,i] + (1-r[b,t,i]) - 1;
subject to S11_5 {i in REQ}:
	(sum {b in BUS, t in 1..T} aux_boarding[b,t,i]) = 1;
subject to S11_6 {i in REQ}:
	(sum {b in BUS, t in 1..T} aux_alighting[b,t,i]) = 1;

#subject to S12 {b in BUS, t in 1..T, i in REQ}:
#	(aux_boarding[b,t,i] * ((sum {j in VERTEX} h[b,t,j] * v_in[i,j]) - 1) + 1) = 1;
subject to S12_1 {b in BUS, t in 1..T, i in REQ, j in VERTEX}:
	aux_pickup_consistent[b,t,i,j] <= (aux_boarding[b,t,i] + h[b,t,j] + V_in[i,j]) / 3;
subject to S12_2 {b in BUS, t in 1..T, i in REQ, j in VERTEX}:
	aux_pickup_consistent[b,t,i,j] >= aux_boarding[b,t,i] + h[b,t,j] + V_in[i,j] - 2;
subject to S12_3 {i in REQ}:
	sum {b in BUS, t in 1..T, j in VERTEX} aux_pickup_consistent[b,t,i,j] = 1;

#subject to S13 {b in BUS, t in 1..T, i in REQ}:
#	(aux_alighting[b,t,i] * (((sum {j in VERTEX} h[b,t,j] * v_out[i,j]) - 1)) + 1) = 1;
subject to S13_1 {b in BUS, t in 1..T, i in REQ, j in VERTEX}:
	aux_leave_consistent[b,t,i,j] <= (aux_alighting[b,t,i] + h[b,t,j] + V_out[i,j]) / 3;
subject to S13_2 {b in BUS, t in 1..T, i in REQ, j in VERTEX}:
	aux_leave_consistent[b,t,i,j] >= aux_alighting[b,t,i] + h[b,t,j] + V_out[i,j] - 2;
subject to S13_3 {i in REQ}:
	sum {b in BUS, t in 1..T, j in VERTEX} aux_leave_consistent[b,t,i,j] = 1;
