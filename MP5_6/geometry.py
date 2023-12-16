# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

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
    x_offset = sorted(list(polygon), key = lambda x: x[0])[-1][0] + 50 # Return largest x value in polygon to offset for line
    # y_offset = sorted(list(polygon), key = lambda x: x[1])[-1][1] + 50 # Return largest y value in polygon to offset for line
    s_x = ( (point[0], point[1]), (point[0] + x_offset, point[1]) )
    # s_y = ( (point[0], point[1]), (point[0], point[1] + y_offset) )
    intersections = 0
    # Edge case, check for if point lies on edges of polygon
    for i in range(-1, 3):
        if point_segment_distance(point, (polygon[i], polygon[i + 1])) == 0:
            return True
    for i in range(-1, 3):
        if do_segments_intersect(s_x, (polygon[i], polygon[i + 1])):
            # Check for weird test cases (bullshit)
            # If the polygon is a vertical straight line, it can mess up the intersection count. 
            if i == -1 and out_of_range_completely(point, polygon):
                intersections = 0
                break
            intersections += 1
            # If both lines are on same y level and not within range, program counts as it as intersecting when it shouldn't.
            if on_same_level(point, polygon, i) and not within_range(point, polygon, i):
                intersections -= 1
    if intersections % 2 == 0:
        return False
    else:
        return True
    
def on_same_level(point, polygon, i):
    return point[1] == polygon[i][1] == polygon[i + 1][1]

def out_of_range_completely(point, polygon):
    return ( (point[0] < polygon[0][0]) and (point[0] < polygon[1][0]) and (point[0] < polygon[2][0]) and (point[0] < polygon[3][0]) ) or ( (point[0] > polygon[0][0]) and (point[0] > polygon[1][0]) and (point[0] > polygon[2][0]) and (point[0] > polygon[3][0]) )

def within_range(point, polygon, i):
    return (polygon[i][0] <= point[0] <= polygon[i + 1][0]) or (polygon[i + 1][0] <= point[0] <= polygon[i][0])

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
        # Straight line path update, change path so that it captures the edges of the sausage to account for condition shift
        path_straight = path
        if not alien.is_circle():
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
        # We need to create a parallelogram modelling the boundaries of the moving sausage       
        if alien.get_shape() == "Horizontal":
            para_inner = (  (centroid[0] - alien_len, centroid[1]),
                            (centroid[0] + alien_len, centroid[1]),
                            (waypoint[0] + alien_len, waypoint[1]),
                            (waypoint[0] - alien_len, waypoint[1])  )
        elif alien.get_shape() == "Vertical":
            para_inner = (  (centroid[0], centroid[1] - alien_len),
                            (centroid[0], centroid[1] + alien_len),
                            (waypoint[0], waypoint[1] + alien_len),
                            (waypoint[0], waypoint[1] - alien_len)  )
            
        for w in walls_format:
            # If wall is close to path, return true
            if segment_distance(path, w) <= radius:
                return True
            
            # We still need to capture extra edge cases of if wall is hitting the ends of the sausage

            # Checking if endpoint of wall is within boundary
            if is_point_in_polygon( w[0], para_inner ) or is_point_in_polygon( w[1], para_inner ):
                return True
            for i in range(-1, 3):
                # Checking if wall is hitting edge of boundary
                if segment_distance( (para_inner[i], para_inner[i + 1]), w ) <= radius:
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

    if la.norm(b-a) == 0: # Catch dividing by 0 error
        d_perp = point_distance(q, a)
    else:
        d_perp = la.norm(np.cross(b-a, a-q))/la.norm(b-a) # Perpendicular distance to line segment
    
    # Distance from endpoints of line segment to point
    d_q_a = point_distance(q, a)
    d_q_b = point_distance(q, b)

    # Range for q in which d_perp is most optimal distance
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
        # Point-slope form, plugging in a and b for y0 and x0 and solving for x
        range_x_q[0] = (q[1] - a[1] + (m_r * a[0])) / m_r
        range_x_q[1] = (q[1] - b[1] + (m_r * b[0])) / m_r
        
    # Sort from least to greatest 
    range_x_q.sort()
    # Check if point is within range of line segment to return d_perp as shortest distance
    if range_x_q[0] <= q[0] and q[0] <= range_x_q[1]:
        return d_perp
    else: # If d_perp is not most optimal distance, most optimal distance has to be the shortest distance to one of the endpoints
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

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
