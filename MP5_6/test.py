import numpy as np
import numpy.linalg as la
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    head, tail = alien.get_head_and_tail()
    centroid = alien.get_centroid()
    radius = alien.get_width()

    # We need to adjust the walls so that they're in the right tuple format
    walls_format = []
    for w in walls:
        walls_format.append(( (w[0], w[1]), (w[2], w[3]) ))

    # Check collision between walls_format and alien in various forms
    if alien.is_circle():
        for w in walls_format:
            if point_segment_distance(centroid, w) <= radius:
                return True
    else:
        for w in walls_format:
            if segment_distance((head, tail), w) <= radius:
                return True
    return False

def create_circle_segments(x, y, r, n):
    # Create n segments from the centroid to the outer edge of the circle
    segments = []
    for i in range(n):
        segments.append((x + np.cos( ( 2*np.pi*i ) / n ), y + np.sin( ( 2*np.pi*i ) / n)))
    return segments

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    head, tail = alien.get_head_and_tail()
    centroid = alien.get_centroid()
    radius = alien.get_width()

    if alien.is_circle():
        min_x = centroid[0] - radius
        min_y = centroid[1] - radius
        max_x = centroid[0] + radius
        max_y = centroid[1] + radius
    else:
        min_x = min(head[0] - radius, tail[0] - radius)
        min_y = min(head[1] - radius, tail[1] - radius)
        max_x = max(head[0] + radius, tail[0] + radius)
        max_y = max(head[1] + radius, tail[1] + radius)

    if min_x > 0 and min_y > 0 and max_x < window[0] and max_y < window[1]:
        return True
    else:
        return False


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    # We can draw a horizontal line from the point and if that line crosses the polygon an odd number of times, it is inside it.
    x_offset = sorted(list(polygon), key = lambda x: x[0])[-1][0] # Return largest x value in polygon to offset for line
    s = ( (point[0], point[1]), (point[0] + x_offset, point[1]) )
    intersections = 0
    # Edge case, check for if point lies on edges of polygon
    for i in range(3):
        if point_segment_distance(point, (polygon[i], polygon[i + 1])) == 0:
            return True
    for i in range(3):
        if do_segments_intersect(s, (polygon[i], polygon[i + 1])):
            intersections += 1
    if intersections % 2 == 0:
        return False
    else:
        return True


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    # Edge case, check if alien is already touching wall at start of path
    if does_alien_touch_wall(alien, walls):
        return True

    head, tail = alien.get_head_and_tail()
    centroid = alien.get_centroid()
    radius = alien.get_width()
    alien_len = alien.get_length() / 2
    path = (centroid, waypoint)

    alien_sausage_vector = np.asarray(head) - np.asarray(tail)
    path_vector = np.asarray(waypoint) - np.asarray(centroid)

    # We need to adjust the walls so that they're in the right tuple format
    walls_format = []
    for w in walls:
        walls_format.append(( (w[0], w[1]), (w[2], w[3]) ))

    # If cross product between sausage vector and path vector is 0, we can treat alien as if it was a moving circle
    if alien.is_circle() or np.cross( alien_sausage_vector, path_vector ) == 0:
        # Straight line path update
        candidates = []
        path_straight = path
        if alien.get_shape() == "Horizontal":
            candidates = [  ( (centroid[0] - alien_len, centroid[1]), (waypoint[0] + alien_len, waypoint[1]) ),
                            ( (centroid[0] + alien_len, centroid[1]), (waypoint[0] - alien_len, waypoint[1]) )  ]
        elif alien.get_shape() == "Vertical":
            candidates = [  ( (centroid[0], centroid[1] - alien_len), (waypoint[0], waypoint[1] + alien_len) ),
                            ( (centroid[0], centroid[1] + alien_len), (waypoint[0], waypoint[1] - alien_len) )  ]
        if point_distance(candidates[0][0], candidates[0][1]) > point_distance(candidates[1][0], candidates[1][1]):
            path_straight = candidates[0]
        else:
            path_straight = candidates[1]
        for w in walls_format:
            if segment_distance(path_straight, w) <= radius:
                return True
    else:
        # We need to create three parallelograms, two defining the outer path, and one defining the inner path
        para_inner = (  (centroid[0] - alien_len, centroid[1]),
                        (centroid[0] + alien_len, centroid[1]),
                        (waypoint[0] + alien_len, waypoint[1]),
                        (waypoint[0] - alien_len, waypoint[1])  )
        
        if alien.get_shape() == "Horizontal":
            para_outer1 = ( (centroid[0], centroid[1]),
                            (centroid[0], centroid[1] + alien_len),
                            (waypoint[0], waypoint[1] + alien_len),
                            (waypoint[0], waypoint[1])  )
            para_outer2 = ( (centroid[0], centroid[1]),
                            (centroid[0], centroid[1] - alien_len),
                            (waypoint[0], waypoint[1] - alien_len),
                            (waypoint[0], waypoint[1])  )
        elif alien.get_shape() == "Vertical":
            para_outer1 = ( (centroid[0], centroid[1]),
                            (centroid[0] + alien_len, centroid[1]),
                            (waypoint[0] + alien_len, waypoint[1]),
                            (waypoint[0], waypoint[1])  )
            para_outer2 = ( (centroid[0], centroid[1]),
                            (centroid[0] - alien_len, centroid[1]),
                            (waypoint[0] - alien_len, waypoint[1]),
                            (waypoint[0], waypoint[1])  )
            
        for w in walls_format:
            if segment_distance(path, w) <= radius + alien_len:
                check1 = False
                check2 = False
                for i in range(3):
                    if do_segments_intersect(w, (para_outer1[i], para_outer1[i + 1])):
                        check1 = True
                    if do_segments_intersect(w, (para_outer2[i], para_outer2[i + 1])):
                        check2 = True
                if (is_point_in_polygon(w[0], para_outer1) or is_point_in_polygon(w[1], para_outer1) or check1):
                    if segment_distance(path, w) <= radius:
                        return True
                elif (is_point_in_polygon(w[0], para_outer2) or is_point_in_polygon(w[1], para_outer2) or check2):
                    if segment_distance(path, w) <= radius:
                        return True
                else:
                    return True
                
    return False

def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    m_inf = False # Checks if slope is infinity
    try:
        m = (s[1][1] - s[0][1]) / (s[1][0] - s[0][0]) # Slope of line segment
    except ZeroDivisionError: # Avoid division by 0
        if s[1][1] - s[0][1] > 0:
            m = 1
        else:
            m = -1
        m_inf = True
    # Calculate reciprocal of m
    m_r_inf = False
    try:
        m_r = -1/m
    except ZeroDivisionError:
        m_r = 0
        m_r_inf = True
    a = np.asarray(s[0])
    b = np.asarray(s[1])
    q = np.asarray(p)
    d_perp = la.norm(np.cross(b-a, a-q))/la.norm(b-a) # Perpendicular distance to line segment
    
    # Distance from endpoints of line segment to point
    d_q_a = point_distance(q, a)
    d_q_b = point_distance(q, b)

    range_x_q = [0, 0]

    # Edge case if slope is infinite
    if m_inf:
        if (a[1] < q[1] and q[1] < b[1]) or (b[1] < q[1] and q[1] < a[1]):
            return d_perp
        else:
            return min(d_q_a, d_q_b)
    elif m_r_inf:
        range_x_q = [a[0], b[0]]
    else:
        range_x_q[0] = (q[1] - a[1] + (m_r * a[0])) / m_r
        range_x_q[1] = (q[1] - b[1] + (m_r * b[0])) / m_r
    
    range_x_q.sort()
    # Check if point is within range of line segment to return d_perp as shortest distance
    if range_x_q[0] <= q[0] and q[0] <= range_x_q[1]:
        return d_perp
    else:
        return min(d_q_a, d_q_b)
    
def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    # Find orientations between both line segments
    ori1 = orientation(s2[0], s1) 
    ori2 = orientation(s2[1], s1) 
    ori3 = orientation(s1[0], s2) 
    ori4 = orientation(s1[1], s2) 
  
    # If both orientations are different, lines intersect
    if ((ori1 != ori2) and (ori3 != ori4)): 
        return True
  
    # s1 and s2[0] are collinear and s2[0] lies on segment s1
    if ((ori1 == 0) and on_segment(s2[0], s1)): 
        return True
  
    # s1 and s2[1] are collinear and s2[1] lies on segment s1
    if ((ori2 == 0) and on_segment(s2[1], s1)): 
        return True
  
    # s2 and s1[0] are collinear and s1[0] lies on segment s2
    if ((ori3 == 0) and on_segment(s1[0], s2)): 
        return True
  
    # s2 and s1[1] are collinear and s1[1] lies on segment s2
    if ((ori4 == 0) and on_segment(s1[1], s2)): 
        return True
  
    # If none of the cases 
    return False
     
def on_segment(p, s):
    # Find if point p lies on line segment s 
    if ( (p[0] <= max(s[0][0], s[1][0])) and (p[0] >= min(s[0][0], s[1][0])) and 
           (p[1] <= max(s[0][1], s[1][1])) and (p[1] >= min(s[0][1], s[1][1])) ): 
        return True
    else:
        return False
  
def orientation(p, s): 
    # Calculate orientation between point and line segment
    ori = (float(p[1] - s[0][1]) * (s[1][0] - p[0])) - (float(p[0] - s[0][0]) * (s[1][1] - p[1]))
    
    # Clockwise orientation 
    if (ori > 0): 
        return 1
    # Counterclockwise orientation
    elif (ori < 0): 
        return 2
    # Collinear orientation 
    else: 
        return 0

def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    # If segments intersect, no distance between points
    if do_segments_intersect(s1, s2):
        return 0
    else:
        d1 = point_segment_distance(s2[0], s1)
        d2 = point_segment_distance(s2[1], s1)
        d3 = point_segment_distance(s1[0], s2)
        d4 = point_segment_distance(s1[1], s2)
        return min(d1, d2, d3, d4)

walls = [(0, 100, 100, 100), (0, 140, 100, 140), (100, 100, 140, 110), (100, 140, 140, 130), (140, 110, 175, 70), (140, 130, 200, 130), (200, 130, 200, 10), (200, 10, 140, 10), (175, 70, 140, 70), (140, 70, 130, 55), (140, 10, 130, 25), (130, 55, 90, 55), (130, 25, 90, 25), (90, 55, 90, 25), (50, 119, 50, 121)]
alien = Alien((30, 120), [20, 0, 20], [11, 20, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', (220, 200))
waypoint = (30, 110)
print( does_alien_path_touch_wall(alien, walls, waypoint) )