#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""

class Trajectory:

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
            desired duration of the trajectory in seconds 
        """
        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.
        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0], 
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0], 
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0], 
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0], 
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0], 
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0], 
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

class LinearTrajectory(Trajectory):

    def __init__(self, total_time, start_position, end_position):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        Trajectory.__init__(self, total_time)
        self.total_time = total_time
        self.start_position = start_position
        self.end_position = end_position

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        return [self.x_func0(time), self.x_func1(time), self.x_func2(time), 0, 1, 0, 0]

    def x_func0(self, time):
        k = (self.end_position[0] + self.start_position[0])/2.0
        m = 8*(k - self.start_position[0])/(self.total_time**2)
        if time <= self.total_time/2:
            return (0.5*m*(time**2)) + self.start_position[0]
        else:
            return (-0.5*m*(time**2)) + m*self.total_time*time + k - ((3/8.0)*m*(self.total_time**2))
            # return (-0.5*m*(time**2)) + m*self.total_time*time + k + ((1/8.0)*m*(self.total_time**2)) - m*(self.total_time**2)/2.0
    
    def x_func1(self, time):
        k = (self.end_position[1] + self.start_position[1])/2.0
        m = 8*(k - self.start_position[1])/(self.total_time**2)
        if time <= self.total_time/2:
            return (0.5*m*(time**2)) + self.start_position[1]
        else:
            return (-0.5*m*(time**2)) + m*self.total_time*time + k - ((3/8.0)*m*(self.total_time**2))
            # return (-0.5*m*(time**2)) + m*self.total_time*time + k + ((1/8.0)*m*(self.total_time**2)) - m*(self.total_time**2)/2.0
    
    def x_func2(self, time):
        k = (self.end_position[2] + self.start_position[2])/2.0
        m = 8*(k - self.start_position[2])/(self.total_time**2)
        if time <= self.total_time/2:
            return (0.5*m*(time**2)) + self.start_position[2]
        else:
            return (-0.5*m*(time**2)) + m*self.total_time*time + k - ((3/8.0)*m*(self.total_time**2))
            # return (-0.5*m*(time**2)) + m*self.total_time*time + k + ((1/8.0)*m*(self.total_time**2)) - m*(self.total_time**2)/2.0

    def v_func0(self, time):
        k = (self.end_position[0] + self.start_position[0])/2.0
        m = 8*(k - self.start_position[0])/(self.total_time**2)
        if time <= self.total_time/2:
            return m*time
        else:
            return -m*time + m*self.total_time
    
    def v_func1(self, time):
        k = (self.end_position[1] + self.start_position[1])/2.0
        m = 8*(k - self.start_position[1])/(self.total_time**2)
        if time <= self.total_time/2:
            return m*time
        else:
            return -m*time + m*self.total_time
    
    def v_func2(self, time):
        k = (self.end_position[2] + self.start_position[2])/2.0
        m = 8*(k - self.start_position[2])/(self.total_time**2)
        if time <= self.total_time/2:
            return m*time
        else:
            return -m*time + m*self.total_time

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        return [self.v_func0(time), self.v_func1(time), self.v_func2(time), 0, 0, 0]

class CircularTrajectory(Trajectory):

    def __init__(self, center_position, radius, total_time):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        #thetadot = sin(x)
        self.center = center_position
        self.radius = radius
        self.total_time = total_time
        self.position = [self.center[1] + self.radius, 0, 0]
        self.m = 8*np.pi/(total_time**2)
        self.C = (2/total_time)*(np.pi + (3*self.m*(total_time**2)/8))
        Trajectory.__init__(self, total_time)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        #self.position = [self.radius*np.cos(time), self.radius*np.sin(time), 0]
        #return [self.position[0], self.position[1], 0, 0, 1, 0, 0]
        #self.theta = self.omega*time
        self.position = [self.center[0] + self.radius*np.cos(self.theta_func(time)), self.center[1] + self.radius*np.sin(self.theta_func(time))]
        return [self.position[0], self.position[1], 0, 0, 1, 0, 0]

    def theta_func(self, time):
        if time <= self.total_time/2:
            return self.m*(time**2)/2
        else:
            return (self.C*time) - (self.m*(time**2)/2)
    
    def thetadot(self, time):
        if time <= self.total_time/2:
            return self.m*time
        else:
            return self.C - (self.m*time)

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        #self.alpha = np.arctan2(self.position[1], self.positiosn[0])
        xdot = -self.radius*np.sin(self.theta_func(time))*self.thetadot(time)
        ydot = self.radius*np.cos(self.theta_func(time))*self.thetadot(time)
        return [xdot, ydot, 0, 0, 0, 1]

class PolygonalTrajectory(Trajectory):
    def __init__(self, points, total_time, first, second, third):
        """
        Remember to call the constructor of Trajectory.
        You may wish to reuse other trajectories previously defined in this file.

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit

        """
        Trajectory.__init__(self, total_time)
        self.points = points
        self.total_time = total_time
        self.trajectory1 = LinearTrajectory(int(self.total_time/self.points), first, second)
        self.trajectory2 = LinearTrajectory(int(self.total_time/self.points), second, third)
        self.trajectory3 = LinearTrajectory(int(self.total_time/self.points), third, first)
        self.current = self.trajectory1
        self.factor = 1

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        self.choose_t(time)
        return self.current.target_pose(self.time)

    def choose_t(self, time):
        if time <= self.total_time/3:
            self.current = self.trajectory1
            self.time = time
        elif time <= 2*self.total_time/3:
            self.current = self.trajectory2
            self.time = time - self.total_time/3
        else:
            self.current = self.trajectory3
            self.time = time - 2*self.total_time/3

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        self.choose_t(time)
        return self.current.target_velocity(self.time)

def define_trajectories(args):
    """ Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        trajectory = LinearTrajectory(3, [0,0,0], [5,0,0])
    elif args.task == 'circle':
        trajectory = CircularTrajectory([0, 0, 0], 2, 4)
    elif args.task == 'polygon':
        trajectory = PolygonalTrajectory(4, 8)
    return trajectory

if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the 
    animate function to visualize the full trajectory in a 3D plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('--animate', action='store_true', help=
        'If you set this flag, the animated trajectory will be shown.'
    )
    args = parser.parse_args()

    trajectory = define_trajectories(args)
    
    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)