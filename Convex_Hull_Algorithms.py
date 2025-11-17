###Question 1

#Given a set of n points S={x1, x2, ..., xn} the convex set will be any set that contains all of these points. 
#The convex hull will be the set which has minimum length between the end points. 
#A convex set would look like [a, b] where a<=minS and b>=maxS. 
#Therefore the convex hull is the convex set which maximises a and minimises b which is the set [minS, maxS]. 
#An algorithm to find the convex hull of a 1d set would need to pass through every element once in order to determine which is the maximum and which is the minimum. 
#Therefore it would have order O(n).

###Question 2

#For each ni assign the point pi = (ni, ni^2). 
#This gives a unique point for each ni on the convex parabola y=x^2 and the convex hull of this will include all of the points pi. 
#The x coordinates of the ordered hull sorted counterclockwise will be the sorted set S. 
#Assigning the ni to points has time complexity O(n) as it just needs to pass through each ni once to assign it. 
#Let the process of computing the convex hull take F(n) time. 
#Therefore the time complexity of sorting these points takes O(n)+F(n). 
#The minimum time complexity for sorting points take O(nlogn) time. 
#Therefore a lower bound for O(n)+F(n) is O(nlogn). 
#This implies that the minimal runtime of computing the convex hull is O(nlogn). 

###Question 3

import math
import numpy as np
import matplotlib.pyplot as plt
import random

def find_angle(a, b): #find angle bewteen ab and the x axis
    ab = [b[0]-a[0], b[1]-a[1]] # vector from a to b

    norm = math.sqrt(ab[0]**2+ab[1]**2) 

    angle = math.acos(ab[0]/norm) # uses same formula as last question

    if ab[1]<0: # checks if vector points down at all meaning that the angle will be clockwise from the positive x axis which we dont want
        angle = 2*math.pi - angle

    return angle, norm

def check_right_turn(x, y, c, sorted_points, i):
    xy = np.array([y[0]-x[0], y[1]-x[1], 0])
    xc = np.array([c[0]-x[0], c[1]-x[1], 0])

    cross = np.cross(xy, xc)

    finished = False

    if i==len(sorted_points)-1:
        finished = True

    elif cross[2]<0: # this means y is not in the convex hull
        sorted_points.remove(y)
        i -= 1
        x = sorted_points[i-2]
        y = sorted_points[i-1]
        c = sorted_points[i]

    elif cross[2]>=0: # this means y is in the convex hull
        x = sorted_points[i-1]
        y = sorted_points[i]
        c = sorted_points[i+1]
        i=i+1

    return x, y, c, sorted_points, i, finished

def graham_scan(S): # S is a list of coordinates of points
    S = list(dict.fromkeys(S)) # this removes any duplicated points as the hull will still look the same and this will prevent bugs

    if len(S)<2: return S
    # Step 1: find base point

    # Find the point with the lowest y (and lowest x if tie)
    b = min(S, key=lambda p: (p[1], p[0]))

    # Step 2: find angles between b and all other points. 
    # u.v = |u||v|cos(theta) rearranges to cos(theta) = u.v/|u||v|. 
    # If u = vector from b to p =(dx, dy) and v = (1,0) then this becomes cos(theta) = dx/sqrt(dx^2+dy^2). 
    # We then need to use arccos and check if the y level of the current point is above or below the y level of the point being compared. 
    # If the y level is below the current point being checked then we need to add 180 degrees to the angle as arccos will give us the smallest angle. 

    angle_to_point = {}

    for p in S:
        if p == b:
            continue
        angle, norm = find_angle(b, p)
        distance = norm ** 2

        if angle not in angle_to_point or distance > angle_to_point[angle][0]:
            angle_to_point[angle] = (distance, p)

    # Step 3: sort the angles, breaking any ties with the shortest distance

    angle_and_point_pairs = [(angle, dist, pt) for angle, (dist, pt) in angle_to_point.items()]
    angle_and_point_pairs.sort()  

    sorted_points = [p for (angle, distance, p) in angle_and_point_pairs]

    P = [b] + sorted_points + [b]

    # Step 4: initialise the convex hull with the first point

    next_point = P[1]

    # Step 5: go through each point c in P and check if xyc forms a left turn or right turn. 
    # If it is a right turn then y is not part of the convex hull and should be removed from consideration

    x = b
    y = next_point
    c = P[2]
    i = 2
    finished = False

    while not finished:

        x, y, c, P, i, finished = check_right_turn(x, y, c, P, i)
        
    return P

def plot_hull(S, hull):
    xhullcoords, yhullcoords = zip(*hull)
    xcoords, ycoords = zip(*S)

    plt.plot(xcoords, ycoords, 'o', color = "black")
    plt.plot(xhullcoords, yhullcoords, color = 'blue')
    plt.show()

S = ((0,0),(1,0),(1,1),(0,1),(0.5,0.5),(0.25,0),(1,0.25),(0.75,1),(0,0.75))

hull = graham_scan(S)

plot_hull(S, hull)

###Question 4

for j in range(10):
    S = []

    for i in range(50):
        S.append((random.random(), random.random()))

    hull = graham_scan(S)

    plot_hull(S, hull)

###Question 5

#Finding the base point requires a linear scan of all n points so it has time complexity O(n). 
#Finding the angles and distances of all n-1 points to the base point requires n-1 calculations each. 
#Each of these caluclations has time complexity O(1) therefore this step has time complexity O(n). 
#Then we need to sort the points by increasing angle. 
#With an efficient sorting algorithm such as the mergesort this has time complexity O(nlogn). 
#Constructing the hull goes linearly through each point checking if it has a left or right turn so has time complexity O(n). 
#Therefore the runtime is dominated by the sorting set as the O(n) steps are insignificant compared to O(nlogn). 
#Therefore the total runtime simplifies to O(nlogn). 

###Question 6

def find_hull(S):
    if len(S) <= 3: return S + [S[0]] # if it is a triangle or smaller this is trivial and is just the set S along with the first coordinate for finishing the loop

    S = list(dict.fromkeys(S)) # this removes any duplicated points as the hull will still look the same and this will prevent bugs
    xs = [p[0] for p in S] # list of x coordinates of S
    if min(xs) == max(xs): # this checks if it is just a vertical line and deals with this separately to prevent bugs
        low = min(S, key=lambda p: p[1])
        high = max(S, key=lambda p: p[1])
        return [low, high] if low != high else [low] # this is just the 1D case 

    Sx = sorted(S, key=lambda p: (p[0], p[1])) # sorts S by x values and breaks ties by sorting by y
    x0 = Sx[len(Sx) // 2][0] # finds the x coordinate of the midpoint

    # Splits the sorted list into left of the middle x value and right of the middle x value and deals with points in the middle seperately. 
    # This ensures that there is never any points across hulls and a vertical line can be drawn between them, specifically on the line x = x0
    left = [p for p in Sx if p[0] < x0] 
    right = [p for p in Sx if p[0] > x0]
    middle = [p for p in Sx if p[0] == x0]

    # Assign middle points to the smaller set (or left if equal)
    if len(left) <= len(right):
        left += middle
    else:
        right += middle

    return merge_hulls(find_hull(left), find_hull(right)) # recursively call the function merge_hulls to merge all of the hulls together

def merge_hulls(A, B): #A and B will be sorted convex hulls from left most point to right
    # these deal with the cases where one of the inputs is an empty list
    if len(A)==0 and len(B)==0:
        return A
    elif len(A)==0:
        beta = min(B, key=lambda p: (p[0], p[1]))
        B_cw = sort_cw(beta, B)
        return B_cw + [B_cw[0]]
    elif len(B)==0:
        alpha = max(A, key=lambda p: (p[0], -p[1]))
        A_cw = sort_cw(alpha, A)
        return A_cw + [A_cw[0]]
    
    # this is the main part of the function which merges two convex hulls 
    alpha =  max(A, key=lambda p: (p[0], -p[1]))
    beta = min(B, key=lambda p: (p[0], p[1]))

    #use the sorting functions to sort A and B clockwise and counterclockwise
    B_cw = sort_cw(beta, B)
    B_ccw = sort_ccw(beta, B)
    A_cw = sort_cw(alpha, A)
    A_ccw = sort_ccw(alpha, A)
    # initialise some variables for the coming loop
    found = False
    i=1
    oldalpha = None
    oldbeta = None
    lower_tangent = []

    #this finds the lower tangent

    K = list(A_cw)
    while not found:
        if i%2 == 1: # this is to alternate between fixing alpha and cycling through beta and fixing beta and cycling through alpha
            oldbeta = beta
            beta = check_tangent(alpha, B_ccw, False)
        
        else:
            oldalpha = alpha
            alpha = check_tangent(beta, K, True)
        
        if oldalpha==alpha or oldbeta==beta: # this checks if either alpha or beta stayed the same and if so increases i to move to checking the other one 
            i += 1

        if oldalpha==alpha and oldbeta==beta: # If both alpha and beta have stayed the same after a whole iteration this means that the tangent points have been found. 
            # This adds the current beta and alpha to the lower tangent list and stops the while loop
            lower_tangent.append(beta)
            lower_tangent.append(alpha) 
            found = True
        
    #need to reset all of these variables as they could have been altered in the last part
    alpha = max(A, key=lambda p: (p[0], -p[1]))
    beta = min(B, key=lambda p: (p[0], p[1]))
    found = False
    i=1
    oldalpha = None
    oldbeta = None
    upper_tangent = []

    L = list(B_cw)

    # this finds the upper tangent
    while not found:
        if i%2 == 1:
            oldbeta = beta
            beta = check_tangent(alpha, L, True)

        else:
            oldalpha = alpha
            alpha = check_tangent(beta, A_ccw, False)
            
        if oldalpha==alpha or oldbeta==beta:
            i += 1

        if oldalpha==alpha and oldbeta==beta:
            upper_tangent.append(alpha)
            upper_tangent.append(beta)
            found = True
        
    GOOD_B = get_subset(B_cw, upper_tangent[1], lower_tangent[0])
    GOOD_A = get_subset(A_cw, lower_tangent[1], upper_tangent[0])

    new_hull = GOOD_A + GOOD_B + [GOOD_A[0]]

    new_hull = sort_cw(new_hull[0], new_hull) + [new_hull[0]]

    return new_hull

def get_subset(S, start, end): # function finds the subset of the list S that starts with 'start' and ends with 'end'
 
    start_index = S.index(start)
    end_index = S.index(end)
    
    if start_index <= end_index:
        return S[start_index:end_index+1]
    else: # Wrap around the end of the list
        return S[start_index:] + S[:end_index+1]
    
def check_tangent(x, S, left): #x is the either alpha or beta and will be checked aggainst all other points in the other set S.  
    # The parameter left is a boolean that lets the function know if i should be checking for right turns or left turns.
    # The first few if statements check trivial cases.
    if len(S)==0:
        p = None

    if len(S)==1:
        p = S[0]

    if len(S)>=2:
        finished = False
        while not finished:
            if len(S)>=2:
                p = S[0]
                c = S[1]

                xp = np.array([p[0]-x[0], p[1]-x[1], 0])
                xc = np.array([c[0]-x[0], c[1]-x[1], 0])

                cross = np.cross(xp, xc) # use cross product to check for left and right turns again

                if cross[2] <= 0 and left==False or cross[2] >= 0 and left==True:  
                    if len(S)==2: # if there are two points to check against and it isnt the first then its obviously going to be the other point
                        p=c
                        finished = True
                    else: 
                        S.remove(p) # removes any points that arent possible tangent points
                else:
                    finished = True
            else:
                finished = True

    return p

def sort_ccw(point, S):
    S = list(dict.fromkeys(S)) #remove any possible duplicates being passed in 

    #initialise some angle and distance lists 
    angles, distances = [], []

    for p in S:
        if p != point: # cycles through every point in S that isnt the point passed into the function
            angle, distance = find_angle(point, p) # using the find angle function from the previous question

            angle += math.pi/2 # this alligns the angles with the downwardx vertical as the point is either the furthest left or right of the set

            if angle>2*math.pi:
                angle -= 2*math.pi

            angles.append(angle)
            distances.append(distance)

    angle_distance_point = list(zip(angles, distances, [p for p in S if p != point])) # this combines each point with its corresponding angle and distance into one array
    angle_distance_point.sort()

    S_ccw = [point] + [p for (angle, distance, p) in angle_distance_point] # this extracts all of the points now that theyve been sorted

    return  S_ccw

def sort_cw(point, S):
    S = list(dict.fromkeys(S)) # removes duplicates from list 
    
    angles, distances = [], []

    for p in S:
        if p != point:
            angle, distance = find_angle(point, p)

            angle += math.pi/2 # alligns the angles with the downwards vertical 

            if angle>2*math.pi:
                angle -= 2*math.pi # prevent negative angles

            angle = 2*math.pi - angle # measure from the other way for clockwise

            angles.append(angle)
            distances.append(distance)

    angle_distance_point = list(zip(angles, distances, [p for p in S if p != point]))
    angle_distance_point.sort()

    S_cw = [point]+[p for (angle, distance, p) in angle_distance_point]
    
    return S_cw 

# simple triangle
S = [(0,0),(1,0),(0,1)]
hull = find_hull(S)
plot_hull(S, hull)

# square with point in centre
S = [(0,0),(1,0),(0,1),(1,1),(0.5,0.5)]
hull = find_hull(S)
plot_hull(S, hull)

# colinear points
S = [(0,0),(1,1),(2,2),(3,3)]
hull = find_hull(S)
plot_hull(S, hull)

# duplicate points
S = [(0,0),(1,0),(1,1),(0,1),(0,0),(1,0)]
hull = find_hull(S)
plot_hull(S, hull)

# all points the same
S = [(1,1),(1,1),(1,1)]
hull = find_hull(S)
plot_hull(S, hull)

###Question 7

# random set of points
for j in range(10):
    S = []
    for i in range(50):
        S.append((random.random(), random.random()))

    hull = find_hull(S)

    plot_hull(S, hull)

###Question 8

#We are given Eulers formula V-E+F=2 where V is the number of vertices, E is the number of edges and F is the number of faces. 
#Each face is surrounded by atleast 3 edges and each edge is shared by exactly 2 faces. 
#This implies that 3F<=2E (1) as each face contributes atleast 3 edges but each edge is counted twice. 
#Rearrange (1) for F and input into Eulers formula. 
#This gives V-E+2/3E>=2. 
#V-1/3E>=2. 
#V-2>=1/3E. 
#E<=3V-6. 
#This implies E<3V. 
#Now rearrange Eulers formula for E and input into (1). 
#This gives 3F<=2(V+F-2). 
#This rearranges to F<=2V-4. 
#This implies F<2V.