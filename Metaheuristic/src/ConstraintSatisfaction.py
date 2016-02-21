
import numpy
import pdb
import bisect

from collections import deque

class ConstraintSatisfaction:

    def selectVariable(solutionObj, variables, choosableVars, orderedVars, originalAssignment, timeWindowViolation, currentRoute):
        if timeWindowViolation is not None:
            # if happened a timeWindowViolation, choose the stop with the lowest time window
            originalAssignment = False

            # if it request hasn't boarded yet, then pick it up
            pickupChoice = orderedVars[0] + solutionObj.totalRequests
            if orderedVars[0] < solutionObj.totalRequests and pickupChoice not in currentRoute:
                # print("1")
                # if it can't be chosed then return None (e.g. because of capacity)
                if pickupChoice in choosableVars:
                    idx = choosableVars.index(pickupChoice)
                else:
                    return None,originalAssignment,timeWindowViolation
            else:
                # print("2")
                # pdb.set_trace()
                # unset the flag of violation
                if orderedVars[0] == timeWindowViolation:
                    timeWindowViolation = None
                idx = choosableVars.index(orderedVars[0])

        elif originalAssignment and len(choosableVars) == len(variables):
            # if the route wasn't changed yet and that's the first variable choice, then pick the first variable
            idx = 0
        elif originalAssignment:
            # if it's not the first variable choice and the route is original, then pick the nearest variable in the tree
            diff = numpy.absolute(numpy.array(choosableVars) - variables[0])
            idx = numpy.argmin(diff)
            originalAssignment = False
        else:
            # heristic: pick the lowest in time window
            originalAssignment = False
            idx = None
            for choice in orderedVars:
                pickupChoice = choice + solutionObj.totalRequests
                if choice < solutionObj.totalRequests and pickupChoice not in currentRoute:
                    if pickupChoice in choosableVars:
                        idx = choosableVars.index(pickupChoice)
                        break
                else:
                    idx = choosableVars.index(choice)
                    break
            # if none of the ordered list was possible, take any
            if idx is None:
                idx = choosableVars.index(min(choosableVars))

        pickedVar = choosableVars[idx]
        del choosableVars[idx]
        if pickedVar in orderedVars:
            del orderedVars[orderedVars.index(pickedVar)]
        return pickedVar, originalAssignment, timeWindowViolation

    def filterDomains(domains, constraint):
        # constraint[0] < constraint[1]. Every element of 0 muss find at least one element in 1 to fulfill this
        newDomainLeft = []
        for left in domains[constraint[0]]:
            if bisect.bisect(domains[constraint[1]], left) < len(domains[constraint[1]]):
                newDomainLeft.append(left)
        # Every element of 1 muss find at least one element in 0 to fulfill this
        newDomainRight = []
        for right in domains[constraint[1]]:
            if bisect.bisect_left(domains[constraint[0]], right) > 0:
                newDomainRight.append(right)

        # Consistent if both are non-empty
        consistent = len(newDomainLeft)>0 and len(newDomainRight)>0
        # Return which variables were altered and the new domains
        alteredVars = []
        if len(newDomainLeft) != len(domains[constraint[0]]):
            alteredVars.append(constraint[0])
            domains[constraint[0]] = newDomainLeft
        if len(newDomainRight) != len(domains[constraint[1]]):
            alteredVars.append(constraint[1])
            domains[constraint[1]] = newDomainRight

        return consistent,domains,alteredVars

    def AC3(domains, constraints, pickedVar):
        newDomains = dict(domains)
        consistent = True

        # Gengeral constraints a!=b for all a,b
        assignedValue = newDomains[pickedVar][0]
        for var,dom in newDomains.items():
            if var != pickedVar and assignedValue in dom:
                dom.remove(assignedValue)
                # If the domain is now empty, inconsistency!
                if len(dom) == 0:
                    consistent = False
                    break

        if consistent:
            # Verify every single constraint with 2-consistency check
            queue = deque(constraints)
            while len(queue)>0:
                currentConstr = queue.popleft()
                # remove inconsistencies from the domains
                consistent,newDomains,alteredVars = ConstraintSatisfaction.filterDomains(newDomains, currentConstr)
                if not consistent:
                    break
                # add new domains to analyse
                queue.extend(filter(
                    lambda x: (x[0] in alteredVars or x[1] in alteredVars) and (x is not currentConstr),\
                    constraints))

        return consistent,newDomains

    def satisfyConstraints(solutionObj, variables, domains, constraints, orderedVars,
        currentRoute, level=0, lastLocation=-1, orignalAssignment=True,
        currentCapacity=0, currentTime=0., timeWindowViolation=None):

        newDomains = None
        if currentCapacity >= solutionObj.maxCapacity:
            # exclude choices that would lead to over capacity
            numpVariables = numpy.array(variables)
            numpOrderedVars = numpy.array(orderedVars)
            choosableVars = numpVariables[numpVariables < solutionObj.totalRequests].tolist()
            orderedChoosableVars = numpOrderedVars[numpOrderedVars < solutionObj.totalRequests].tolist()
        else:
            choosableVars = list(variables)
            orderedChoosableVars = list(orderedVars)

        while len(choosableVars)>0:
            # Select a variable (request location) from the list of variables(routes)
            pickedVar, originalAssignment,timeWindowViolation = ConstraintSatisfaction.selectVariable(solutionObj,
                variables, choosableVars, orderedChoosableVars,
                orignalAssignment, timeWindowViolation, currentRoute)

            if pickedVar is not None:
                # Check if assignment doesn't violate time window
                minT,maxT = solutionObj.getWindowOf(pickedVar)
                distance = solutionObj.getTimeDistance(lastLocation,pickedVar)
                waitingTime = solutionObj.getWaitingTime(pickedVar)
                minPossible = currentTime + distance
                newCurrentTime = max(minT, minPossible) + waitingTime
                # tab = ''.join([' ' for i in range(level)])
                # print(tab, pickedVar, newCurrentTime, minT, maxT)
                if newCurrentTime <= maxT:

                    # The assign the value of "level" to this variable
                    domainBackup = domains[pickedVar]
                    domains[pickedVar] = list([level])

                    # Check Arc Consistency of the assignment
                    consistent,consistentDomains = ConstraintSatisfaction.AC3(domains, constraints, pickedVar)
                    if consistent:

                        # if not the end of the tree, call recursively w/ level+1
                        if len(variables) > 1:
                            restVariables = list(variables)
                            restVariables.remove(pickedVar)

                            newOrderedVars = list(orderedVars)
                            newOrderedVars.remove(pickedVar)

                            currentRoute[level] = pickedVar

                            newCapacity = currentCapacity+1 if pickedVar >= solutionObj.totalRequests else currentCapacity-1

                            newDomains,timeWindowViolation = ConstraintSatisfaction.satisfyConstraints(solutionObj,
                                restVariables, consistentDomains,
                                constraints, newOrderedVars, currentRoute,
                                level+1, pickedVar,originalAssignment,
                                newCapacity, newCurrentTime, timeWindowViolation)

                            # if this assignment succeeded, break the loop
                            if newDomains is not None:
                                break
                        else:
                            newDomains = consistentDomains
                            break
                    elif timeWindowViolation is not None:
                        # occured time window violation once, then tried the most constrained, didn't work, return back tracking
                        #print("violation")
                        return None,timeWindowViolation

                    # restore domain to try another variable
                    domains[pickedVar] = domainBackup
                    currentRoute[level] = -1
                else:
                    # return state of timeWindowViolation, which alters the heuristic of picking variables
                    #print("violation")
                    if timeWindowViolation is None:
                        return None,pickedVar
                    else:
                        return None,timeWindowViolation
            else:
                # variable couldn't be chosen because of time window violation, return
                #print("violation")
                return None,timeWindowViolation

        # If the list of variables was exhausted, and no valid assignment was found, return None
        return newDomains,None
