# Convex Hull Algorithms

Two algorithms for finding the convex hull of a set of points in the plane: the smallest convex polygon that contains them all. Written from scratch in Python, with plots of the results.

## The algorithms

**Graham scan** (`graham_scan`)

1. Take the lowest point as the base point.
2. Sort every other point by the angle it makes with the base point.
3. Walk round the points in that order. Whenever three consecutive points make a right turn, the middle one is inside the hull, so drop it.

**Divide and conquer** (`find_hull`)

1. Sort the points by x coordinate and split them into a left half and a right half.
2. Find the hull of each half the same way, recursively. Three points or fewer are their own hull.
3. Join the two hulls (`merge_hulls`): find the upper and lower tangent lines that touch both, then keep only the outer part of each hull between the tangents.

## Using it

```python
from convex_hull import graham_scan, find_hull, plot_hull

points = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]

hull = graham_scan(points)   # or find_hull(points)
plot_hull(points, hull)
```

Both functions take a list of `(x, y)` tuples and return the hull as a closed list of points, with the first point repeated at the end so it can be plotted directly. `graham_scan` returns the hull counterclockwise and `find_hull` clockwise.

To see both algorithms on a few fixed examples and on random point sets:

```bash
pip install numpy matplotlib
python convex_hull.py
```

## Running time

Graham scan is O(n log n). Finding the base point and the angles is O(n), the final walk is O(n) because each point is added and removed at most once, so the sort dominates.

No algorithm can beat O(n log n) in general, because sorting can be turned into a convex hull problem. To sort numbers x1, ..., xn, place each one on the parabola at (xi, xi²). Every one of these points is on the hull, and reading the hull in order gives the numbers sorted. A faster hull algorithm would therefore be a faster sorting algorithm.

## Edge cases

- Duplicate points are removed before the hull is built.
- Points that all lie on one vertical line are handled as a special case in `find_hull`.
- Points lying exactly on an edge of the hull, between two corners, are sometimes kept in the output. The polygon drawn is still the correct hull.

## Built with

Python, NumPy, matplotlib.
