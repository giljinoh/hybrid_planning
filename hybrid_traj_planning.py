"""

Probabilistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interpolate
import matplotlib.pyplot as plt
import test_MTSOS
import numpy as np
#from sympy import Intergral, Symbol, pprint

show_animation = True

"""
Cubic spline planner

Author: Atsushi Sakai(@Atsushi_twi)

"""
import math
import numpy as np
import bisect


class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


class Spline2D:
    """
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s



def approximate_b_spline_path(x: list, y: list, n_path_points: int,
                              degree: int = 3) -> tuple:
    """
    approximate points with a B-Spline path

    :param x: x position list of approximated points
    :param y: y position list of approximated points
    :param n_path_points: number of path points
    :param degree: (Optional) B Spline curve degree
    :return: x and y position list of the result path
    """
    t = range(len(x))
    x_tup = scipy_interpolate.splrep(t, x, k=degree)
    y_tup = scipy_interpolate.splrep(t, y, k=degree)

    x_list = list(x_tup)
    x_list[1] = x + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    y_list[1] = y + [0.0, 0.0, 0.0, 0.0]

    ipl_t = np.linspace(0.0, len(x) - 1, n_path_points)
    rx = scipy_interpolate.splev(ipl_t, x_list)
    ry = scipy_interpolate.splev(ipl_t, y_list)

    return rx, ry


def interpolate_b_spline_path(x: list, y: list, n_path_points: int,
                              degree: int = 3) -> tuple:
    """
    interpolate points with a B-Spline path

    :param x: x positions of interpolated points
    :param y: y positions of interpolated points
    :param n_path_points: number of path points
    :param degree: B-Spline degree
    :return: x and y position list of the result path
    """
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)

    travel = np.linspace(0.0, len(x) - 1, n_path_points)
    return spl_i_x(travel), spl_i_y(travel)

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

class Node:
    """node with properties of g, h, coordinate and parent node"""

    def __init__(self, R=1, G=0, H=0, coordinate=None, parent=None):
        self.R = R
        self.G = G
        self.H = H
        self.F = G + H
        self.parent = parent
        self.coordinate = coordinate

    def reset_f(self):
        self.F = self.G + self.H


def hcost(node_coordinate, goal):
    dx = abs(node_coordinate[0] - goal[0])
    dy = abs(node_coordinate[1] - goal[1])
    hcost = dx + dy
    return hcost


def gcost(fixed_node, update_node_coordinate):
    dx = abs(fixed_node.coordinate[0] - update_node_coordinate[0])
    dy = abs(fixed_node.coordinate[1] - update_node_coordinate[1])
    gc = math.hypot(dx, dy)  # gc = move from fixed_node to update_node
    gcost = fixed_node.G + gc  # gcost = move from start point to update_node
    return gcost

def find_neighbor(node, ob, closed):
    # generate neighbors in certain condition
    ob_list = ob.tolist()
    neighbor: list = []
    #print(node.coordinate)
    for x in range(int(node.coordinate[0]) - 1, int(node.coordinate[0]) + 2):
        for y in range(int(node.coordinate[1]) - 1, int(node.coordinate[1]) + 2):
            if [x, y] not in ob_list:
                # find all possible neighbor nodes
                neighbor.append([x, y])
    # remove node violate the motion rule
    # 1. remove node.coordinate itself
    neighbor.remove(node.coordinate)
    # 2. remove neighbor nodes who cross through two diagonal
    # positioned obstacles since there is no enough space for
    # robot to go through two diagonal positioned obstacles

    # top bottom left right neighbors of node
    top_nei = [node.coordinate[0], node.coordinate[1] + 1]
    bottom_nei = [node.coordinate[0], node.coordinate[1] - 1]
    left_nei = [node.coordinate[0] - 1, node.coordinate[1]]
    right_nei = [node.coordinate[0] + 1, node.coordinate[1]]
    # neighbors in four vertex
    lt_nei = [node.coordinate[0] - 1, node.coordinate[1] + 1]
    rt_nei = [node.coordinate[0] + 1, node.coordinate[1] + 1]
    lb_nei = [node.coordinate[0] - 1, node.coordinate[1] - 1]
    rb_nei = [node.coordinate[0] + 1, node.coordinate[1] - 1]

    # remove the unnecessary neighbors
    if top_nei and left_nei in ob_list and lt_nei in neighbor:
        neighbor.remove(lt_nei)
    if top_nei and right_nei in ob_list and rt_nei in neighbor:
        neighbor.remove(rt_nei)
    if bottom_nei and left_nei in ob_list and lb_nei in neighbor:
        neighbor.remove(lb_nei)
    if bottom_nei and right_nei in ob_list and rb_nei in neighbor:
        neighbor.remove(rb_nei)
    neighbor = [x for x in neighbor if x not in closed]
    return neighbor

def find_path(open_list, closed_list, goal, obstacle):
    # searching for the path, update open and closed list
    # obstacle = obstacle and boundary
    flag = len(open_list)
    for i in range(flag):
        node = open_list[0]
        open_coordinate_list = [node.coordinate for node in open_list]
        closed_coordinate_list = [node.coordinate for node in closed_list]
        temp = find_neighbor(node, obstacle, closed_coordinate_list)
        for element in temp:
            if element in closed_list:
                continue
            elif element in open_coordinate_list:
                # if node in open list, update g value
                ind = open_coordinate_list.index(element)
                new_g = gcost(node, element)
                if new_g <= open_list[ind].G:
                    open_list[ind].G = new_g
                    open_list[ind].reset_f()
                    open_list[ind].parent = node
            else:  # new coordinate, create corresponding node
                ele_node = Node(R=set_radius(element,obstacle),coordinate=element, parent=node,
                                G=gcost(node, element), H=hcost(element, goal))
                open_list.append(ele_node)
        open_list.remove(node)
        closed_list.append(node)
        open_list.sort(key=lambda x: x.F)
    return open_list, closed_list

def boundary_and_obstacles(start, goal, top_vertex, bottom_vertex, ox, oy,obs_number):
    """
    :param start: start coordinate
    :param goal: goal coordinate
    :param top_vertex: top right vertex coordinate of boundary
    :param bottom_vertex: bottom left vertex coordinate of boundary
    :param obs_number: number of obstacles generated in the map
    :return: boundary_obstacle array, obstacle list
    """

    # below can be merged into a rectangle boundary
    #bound = np.vstack((ox, oy)).T.tolist()
    '''
    ay = list(range(bottom_vertex[1], top_vertex[1]))
    ax = [bottom_vertex[0]] * len(ay)
    cy = ay
    cx = [top_vertex[0]] * len(cy)
    bx = list(range(bottom_vertex[0] + 1, top_vertex[0]))
    by = [bottom_vertex[1]] * len(bx)
    dx = [bottom_vertex[0]] + bx + [top_vertex[0]]
    dy = [top_vertex[1]] * len(dx)
    '''


    # generate random obstacles
    ob_x = 0# np.random.randint(bottom_vertex[0] + 1,
               #              top_vertex[0], obs_number).tolist()
    ob_y = 0#np.random.randint(bottom_vertex[1] + 1,
            #                 top_vertex[1], obs_number).tolist()
    # x y coordinate in certain order for boundary
    #x = ax + bx + cx + dx
    #y = ay + by + cy + dy

    obstacle = np.vstack((ob_x, ob_y)).T.tolist()
    #print(obstacle)
    # remove start and goal coordinate in obstacle list
    obstacle = [coor for coor in obstacle if coor != start and coor != goal]
    obs_array = np.array(obstacle)
    bound = np.vstack((ox, oy)).T
    bound_obs = np.vstack((bound, obs_array))
    return bound_obs, obstacle

def node_to_coordinate(node_list):
    # convert node list into coordinate list and array
    coordinate_list = [node.coordinate for node in node_list]
    return coordinate_list

def node_to_radius(node_list):
    # convert node list into coordinate list and array
    radius_list = [node.R for node in node_list]
    return radius_list

def draw(close_origin, close_goal, start, end, bound):
    # plot the map
    if not close_goal.tolist():  # ensure the close_goal not empty
        # in case of the obstacle number is really large (>4500), the
        # origin is very likely blocked at the first search, and then
        # the program is over and the searching from goal to origin
        # will not start, which remain the closed_list for goal == []
        # in order to plot the map, add the end coordinate to array
        close_goal = np.array([end])
    plt.cla()
    plt.gcf().set_size_inches(11, 9, forward=True)
    plt.axis('equal')
    plt.plot(close_origin[:, 0], close_origin[:, 1], 'oy')
    plt.plot(close_goal[:, 0], close_goal[:, 1], 'og')
    plt.plot(bound[:, 0], bound[:, 1], 'sk')
    plt.plot(end[0], end[1], '*b', label='Goal')
    plt.plot(start[0], start[1], '^b', label='Origin')
    plt.legend()
    plt.pause(0.0001)

def check_node_coincide(close_ls1, closed_ls2):
    """
    :param close_ls1: node closed list for searching from start
    :param closed_ls2: node closed list for searching from end
    :return: intersect node list for above two
    """
    # check if node in close_ls1 intersect with node in closed_ls2
    cl1 = node_to_coordinate(close_ls1)
    cl2 = node_to_coordinate(closed_ls2)
    intersect_ls = [node for node in cl1 if node in cl2]
    return intersect_ls

def find_node_index(coordinate, node_list):
    # find node index in the node list via its coordinate
    ind = 0
    for node in node_list:
        if node.coordinate == coordinate:
            target_node = node
            ind = node_list.index(target_node)
            break
    return ind

def get_path(org_list, goal_list, coordinate):
    # get path from start to end
    path_org: list = []
    path_goal: list = []
    ind = find_node_index(coordinate, org_list)
    node = org_list[ind]
    while node != org_list[0]:
        path_org.append(node.coordinate)
        node = node.parent
    path_org.append(org_list[0].coordinate)
    ind = find_node_index(coordinate, goal_list)
    node = goal_list[ind]
    while node != goal_list[0]:
        path_goal.append(node.coordinate)
        node = node.parent
    path_goal.append(goal_list[0].coordinate)
    path_org.reverse()
    path = path_org + path_goal
    path = np.array(path)
    return path

def find_surrounding(coordinate, obstacle):
    # find obstacles around node, help to draw the borderline
    boundary: list = []
    for x in range(coordinate[0] - 1, coordinate[0] + 2):
        for y in range(coordinate[1] - 1, coordinate[1] + 2):
            if [x, y] in obstacle:
                boundary.append([x, y])
    return boundary

def get_border_line(node_closed_ls, obstacle):
    # if no path, find border line which confine goal or robot
    border: list = []
    coordinate_closed_ls = node_to_coordinate(node_closed_ls)
    for coordinate in coordinate_closed_ls:
        temp = find_surrounding(coordinate, obstacle)
        border = border + temp
    border_ary = np.array(border)
    return border_ary

def draw_control(org_closed, goal_closed, flag, start, end, bound, obstacle):
    """
    control the plot process, evaluate if the searching finished
    flag == 0 : draw the searching process and plot path
    flag == 1 or 2 : start or end is blocked, draw the border line
    """
    stop_loop = 0  # stop sign for the searching
    org_closed_ls = node_to_coordinate(org_closed)
    org_array = np.array(org_closed_ls)
    goal_closed_ls = node_to_coordinate(goal_closed)
    goal_array = np.array(goal_closed_ls)
    org_closed_ra = node_to_radius(org_closed)
    org_array_ra = np.array(org_closed_ra)
    goal_closed_ra = node_to_radius(goal_closed)
    goal_array_ra = np.array(goal_closed_ra)
    path = None
    if show_animation:  # draw the searching process
        draw(org_array, goal_array, start, end, bound)
    if flag == 0:
        node_intersect = check_node_coincide(org_closed, goal_closed)
        if node_intersect:  # a path is find
            path = get_path(org_closed, goal_closed, node_intersect[0])
            stop_loop = 1
            print('Path is find!')
            if show_animation:  # draw the path
                plt.plot(path[:, 0], path[:, 1], '-r')
                for i in range(len(path)):
                    draw_circle = plt.Circle((path[i, 0], path[i, 1]), org_array_ra[i], fill=False)
                    plt.gcf().gca().add_artist(draw_circle)
                plt.title('Robot Arrived', size=20, loc='center')
                plt.pause(0.01)
                plt.show()
    elif flag == 1:  # start point blocked first
        stop_loop = 1
        print('There is no path to the goal! Start point is blocked!')
    elif flag == 2:  # end point blocked first
        stop_loop = 1
        print('There is no path to the goal! End point is blocked!')
    if show_animation:  # blocked case, draw the border line
        info = 'There is no path to the goal!' \
               ' Robot&Goal are split by border' \
               ' shown in red \'x\'!'
        if flag == 1:
            border = get_border_line(org_closed, obstacle)
            plt.plot(border[:, 0], border[:, 1], 'xr')
            plt.title(info, size=14, loc='center')
            plt.pause(0.01)
            plt.show()
        elif flag == 2:
            border = get_border_line(goal_closed, obstacle)
            plt.plot(border[:, 0], border[:, 1], 'xr')
            plt.title(info, size=14, loc='center')
            plt.pause(0.01)
            plt.show()
    return stop_loop, path

def set_radius(coor,ob,min=3,max=5):
    # generate neighbors in certain condition
    ob_list = ob.tolist()
    for i in range(len(ob_list)):
        ob_list[i][0] = round(ob_list[i][0],1)
        ob_list[i][1] = round(ob_list[i][1],1)

    #print(ob_list)

    radius_list = []
    radius_can = []
    #print(coor)
    x_list = np.arange(round(coor[0]) - 3, round(coor[0]) + 3, 0.1)
    #print(x_list)
    y_list = np.arange(round(coor[1]) - 3, round(coor[1]) + 3, 0.1)


    # for x in x_list:
    #     for y in y_list:
    for i in range(len(ob_list)):
        if euclidean_distance_2(coor,ob_list[i]) < 5:
            #print('hi')
            radius_can.append(ob_list[i])
    if len(radius_can) == 0:
        return max

    radius = euclidean_distance(coor,radius_can)
    #print('radius',radius_can)
    if radius < min:
        return min
    elif radius > max:
        return max
    else:
        return radius

def euclidean_distance(node, pos):
  distance = []
  for i in range(len(pos)):
    distance.append((((pos[i][0] - node[0]) ** 2) + ((pos[i][1] - node[1]) ** 2)) ** 0.5)
  #print(node)
  #print(distance)
  if len(distance) == 0:
    return 1
  else:
    return min(distance)

def euclidean_distance_2(coor1, coor2):
  distance = ((((coor1[0] - coor2[0]) ** 2) + ((coor1[1] - coor2[1]) ** 2)) ** 0.5)
  #print(distance)
  return distance


def searching_control(start, end, bound, obstacle):
    """manage the searching process, start searching from two side"""
    # initial origin node and end node
    origin = Node(R=set_radius(start,bound),coordinate=start, H=hcost(start, end))
    goal = Node(R=set_radius(end,bound),coordinate=end, H=hcost(end, start))
    # list for searching from origin to goal
    origin_open: list = [origin]
    origin_close: list = []
    # list for searching from goal to origin
    goal_open = [goal]
    goal_close: list = []
    # initial target
    target_goal = end
    # flag = 0 (not blocked) 1 (start point blocked) 2 (end point blocked)
    flag = 0  # init flag
    path = None
    while True:
        # searching from start to end
        origin_open, origin_close = \
            find_path(origin_open, origin_close, target_goal, bound)
        if not origin_open:  # no path condition
            flag = 1  # origin node is blocked
            draw_control(origin_close, goal_close, flag, start, end, bound,
                         obstacle)
            break
        # update target for searching from end to start
        target_origin = min(origin_open, key=lambda x: x.F).coordinate

        # searching from end to start
        goal_open, goal_close = \
            find_path(goal_open, goal_close, target_origin, bound)
        if not goal_open:  # no path condition
            flag = 2  # goal is blocked
            draw_control(origin_close, goal_close, flag, start, end, bound,
                         obstacle)
            break
        # update target for searching from start to end
        target_goal = min(goal_open, key=lambda x: x.F).coordinate

        # continue searching, draw the process
        stop_sign, path = draw_control(origin_close, goal_close, flag, start,
                                       end, bound, obstacle)
        if stop_sign:
            break
    return path

def GetNewCircleCenter(p1,p2,p3):
    global grid_size
    #y = ax+b          -ax + y = b
    if p3[1]-p1[1] == 0 or p3[0]-p1[0] == 0:
        return p2
    else:
        line_1_3_inclination = (p3[1]-p1[1])/(p3[0]-p1[0])
    b1 = p1[1] - (line_1_3_inclination * p1[0])
    b2 = p2[1] - (-1 / line_1_3_inclination * p2[0])

    # Ax = B

    A = np.array([[-line_1_3_inclination, 1], [1 / line_1_3_inclination, 1]])
    B = np.array([b1,b2])
    C = np.linalg.solve(A,B)
    smooth_point = [(C[0]+p2[0])/2,(C[1]+p2[1])/2]
    #print('circle',C)
    if euclidean_distance_2(p2,C) < 3 :#grid_size:
        return smooth_point
    return C


def eval(path):
    J=0
    for i in range(1,len(path)-2):
        a = euclidean_distance_2(path[i-1],path[i])
        b = euclidean_distance_2(path[i],GetNewCircleCenter(path[i-1],path[i],path[i+1]))
        c = euclidean_distance_2(path[i-1],GetNewCircleCenter(path[i-1],path[i],path[i+1]))

        s = (a+b+c)/2
        k = (s*(s-a)*(s-b)*(s-c))**0.5
        if a*b*c == 0:
            k_i = 3
        else:
            k_i =  (4*k)/(a*b*c)

        a1 = euclidean_distance_2(path[i], path[i+1])
        b1 = euclidean_distance_2(path[i+1], GetNewCircleCenter(path[i], path[i+1], path[i + 2]))
        c1 = euclidean_distance_2(path[i], GetNewCircleCenter(path[i], path[i+1], path[i + 2]))

        s1 = (a1 + b1 + c1) / 2
        k1 = (s1*(s1 - a1)*(s1 - b1)*(s1 - c1)) ** 0.5
        if a1*b1*c1 == 0:
            k_i1 = 3
        else:
            k_i1 =  (4*k1)/(a1*b1*c1)


        J += (((k_i**2)+(k_i1**2)) * euclidean_distance_2(path[i-1],path[i])) / 2

    #print('J',J)
    return J


def main(obstacle_number = 0):
    global grid_size
    print(__file__ + " start!!")



    ox = []
    oy = []


    x = [0.0, 20.0, 30.0, 40.0, 60.0]
    y = [0.0, 4.0, 12.0, 20.0, 24.0]
    # x = [0.0, 9.0, 29.0, 56.0, 99.0]
    # y = [0.0, 41.161, 70.42, 89.8, 99.0]
    ds = 0.1  # [m] distance of each intepolated points

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)


    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        ox.append(ix)
        oy.append(iy)


    x_ = [0.0, 20.0, 30.0, 40.0, 60.0]
    y_ = [10.0, 14.0, 22.0, 30.0, 34.0]
    # x = [16.333, 22.0, 27.0, 67.0, 99.0]
    # y = [0.0, 30.265, 55.055, 76.883, 83.606]
    ds_ = 0.1  # [m] distance of each intepolated points

    sp = Spline2D(x_, y_)
    s = np.arange(0, sp.s[-1], ds_)

    # start and goal position
    sx = (x[0]+x_[0])/2.0 #8.0  # [m]
    sy = (y[0]+y_[0])/2.0 #0.0  # [m]
    gx = (x[-1]+x_[-1])/2.0 - 1 # 99.0  # [m]
    gy = (y[-1]+y_[-1])/2.0 # 90.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 3.0  # [m]
    # robot_size = 5.0  # [m]
    start = [sx, sy]
    end = [gx, gy]

    top_vertex = [100, 100]  # top right vertex of boundary
    bottom_vertex = [0, 0]  # bottom left vertex of boundary

    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        ox.append(ix)
        oy.append(iy)




    # for i in range(top_vertex[0]):
    #     ox.append(i)
    #     oy.append(bottom_vertex[1])
    # for i in range(top_vertex[1]):
    #     ox.append(top_vertex[0])
    #     oy.append(i)
    # for i in range(top_vertex[0]):
    #     ox.append(i)
    #     oy.append(top_vertex[1])
    # for i in range(top_vertex[1]):
    #     ox.append(bottom_vertex[1])
    #     oy.append(i)
    # for i in range(int(top_vertex[1]/3*2)):
    #     ox.append(top_vertex[0]/3)
    #     oy.append(i)
    # for i in range(int(top_vertex[1]/3*2)):
    #     ox.append(top_vertex[0]/3*2)
    #     oy.append(top_vertex[1] - i)

    bound, obstacle = boundary_and_obstacles(start, end, top_vertex,
                                             bottom_vertex, ox, oy,
                                             obstacle_number)

    obstacle_x = [obstacle[i][0] for i in range(len(obstacle))]
    obstacle_y = [obstacle[i][1] for i in range(len(obstacle))]
    bound_x = [bound[i][0] for i in range(len(bound))]
    bound_y = [bound[i][1] for i in range(len(bound))]



    #path = searching_control(start, end, bound, obstacle)



    if show_animation:

        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        #plt.plot(path[:,0], path[:,1], "-y")
        plt.plot(bound_x,bound_y,"bo")
        #for i in range(len(path)):
         #   draw_circle = plt.Circle((path[i,0], path[i,1]), 3,fill=False)
          #  plt.gcf().gca().add_artist(draw_circle)
        plt.grid(True)
        plt.axis("equal")


    a_star = AStarPlanner(bound_x, bound_y, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    path_arr = np.vstack((rx, ry)).T.tolist()



    smooth_path = []
    for i in range(1, len(path_arr) - 1):
        smooth_path.append(GetNewCircleCenter(path_arr[i - 1], path_arr[i], path_arr[i + 1]))

    smooth_path_x = [smooth_path[i][0] for i in range(len(smooth_path))]
    smooth_path_y = [smooth_path[i][1] for i in range(len(smooth_path))]

    print('1',eval(path_arr))
    #print('2',eval(smooth_path))
    rx1, ry1 = approximate_b_spline_path(smooth_path_x,smooth_path_y,100)
    #rx1, ry1 = approximate_b_spline_path(rx, ry, 20)
    s_path_arr = np.vstack((rx1, ry1)).T.tolist()
    #print('3',eval(s_path_arr))
    # print(s_path_arr)
    arr=[]
    for i in range(len(s_path_arr)):
        arr.append(s_path_arr[i][0])
        arr.append(s_path_arr[i][1])
    print(arr , len(arr))

    rad_list = []
    for i in range(len(s_path_arr)):
        #rad_list.append(set_radius(path_arr[i], bound))
        rad_list.append(set_radius(s_path_arr[i], bound))

    if show_animation:
        #plt.plot(rx, ry, "-r")
        #plt.plot(smooth_path_x, smooth_path_y, "-r")
        plt.plot(rx1, ry1, "-r")
        for i in range(len(s_path_arr)):
            #draw_circle = plt.Circle((path_arr[i][0], path_arr[i][1]), rad_list[i],fill=False)
            draw_circle = plt.Circle((s_path_arr[i][0], s_path_arr[i][1]), rad_list[i], fill=False)
            plt.gcf().gca().add_artist(draw_circle)
        plt.pause(0.001)
        test_MTSOS.main(arr)
        plt.show()



if __name__ == '__main__':
    main(obstacle_number=0)
