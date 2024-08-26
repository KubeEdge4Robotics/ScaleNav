#!/usr/bin/env python3
#! This .py file must be run in ROS environment.
import math
import sys
import os
sys.path.append('../')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cv2
import rospy
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import tf
from sensor_msgs.msg import Image, CompressedImage
import geometry_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
from math import pi, radians, sqrt, pow, atan2
import pickle
import numpy as np

from dataset.dataset_utils import _norm_heading, _rotate_coord
from utils.realsense import RealSenseCamera


def _calc_distance(dx, dy):
    return math.sqrt(pow(dx, 2) + pow(dy, 2))

def _quart_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def _quat_to_angle(x, y, z, w):
    _, _, yaw = _quart_to_rpy(x, y, z, w)
    return yaw


class ROSInterface:
    def __init__(self, args):
        if args.no_ros:
            return
        '''node init'''
        rospy.init_node('deploy', anonymous=False)
        rospy.on_shutdown(self.shutdown)

        rate = args.rate  # 30
        self.rate = rospy.Rate(rate)

        '''topic names'''
        self.vel_topic_name = '/smoother_cmd_vel'
        self.camera_topic_name = "/camera/color/image_raw/compressed"
        self.scan_topic_name = '/scan'
        self.odom_frame = '/odom'
        rospy.loginfo(
            f'topics:{self.vel_topic_name}, {self.camera_topic_name}, {self.scan_topic_name}')
        rospy.loginfo(f'frame:{self.odom_frame}')

        '''hyperparams'''
        self.linear_speed = 0.4  # *
        self.angular_speed = 0.1  # *
        self.angular_tolerance = 0  # * key params to tune: radians(2.5)=0.2
        self.distance_tolerance = 0.03
        self.slowdown_thresh = 0.2
        self.slow_speed = 0.2
        self.sleep_t = 0.02
        self.duration = 5.0
        self.lidar_detect_distance = 0.1  # *
        self.scan_error = 0.2
        self.can_go = True
        self.store_interval = 2 * (1 / self.sleep_t)
        rospy.loginfo('angular_tolerance:{}'.format(self.angular_tolerance))

        '''publisher and subscriber'''
        self.cmd_vel = rospy.Publisher(
            self.vel_topic_name, Twist, queue_size=5)
        self.scan_sub = rospy.Subscriber(
            self.scan_topic_name, LaserScan, self.get_scan, queue_size=10)

        self.bridge = CvBridge()  # Instantiate CvBridge
        # self.cv2_img = CompressedImage()
        # self.camera_sub = rospy.Subscriber(self.camera_topic_name, CompressedImage, self.get_image, queue_size=50)
        self.camera_sub = RealSenseCamera()

        '''initialize tf listener and set odom_frame and base_frame'''
        self.tf_listener = tf.TransformListener()
        rospy.sleep(self.sleep_t)  # cache tf
        try:
            self.tf_listener.waitForTransform(self.odom_frame,
                                              '/base_link',
                                              rospy.Time(),
                                              rospy.Duration(self.duration))
            self.base_frame = '/base_link'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException) as e:
            rospy.loginfo("Cannot find base_frame transformed from /odom")
            rospy.loginfo(e)

        '''record last goal and error'''
        self.pre_rotation = 0
        self.pre_rotation_error = 0
        self.pre_position = [0, 0]
        self.pre_posistion_error = [0, 0]
        self.pre_dx, self.pre_dy = 1, 1
        self.sign = 1

        '''maintain a buffer between two adjacent waypoints preparing for backtracking'''
        self._buffer = []

    def get_real_time_img(self):
        # return self.cv2_img  # use subscribing topic
        color_img, _, _ = self.camera_sub.get_image(show=True)
        return color_img

    def get_real_time_pose(self):
        return self.get_odom()

    def send_action(self, action):
        linear_v, angular_v = action[0], action[1]
        move_cmd = Twist()
        move_cmd.linear.x = linear_v
        move_cmd.angular.z = angular_v
        now = rospy.Time.now()
        while rospy.Time.now() < now + rospy.Duration.from_sec(2.0):
            self.cmd_vel.publish(move_cmd)
            self.rate.sleep()
        
    
    def rotate(self, goal_angle):
        rospy.loginfo("rotation")
        _, _, rotation = self.get_odom()
        move_cmd = Twist()
        move_cmd.angular.z = self.angular_speed if goal_angle > 0 else -self.angular_speed
        pre_angle = rotation
        acc_angle_turn = 0
        t = 0
        while abs(goal_angle) - abs(acc_angle_turn) > self.angular_tolerance \
                and not rospy.is_shutdown():
            self.cmd_vel.publish(move_cmd)
            self.rate.sleep()
            if t % self.store_interval == 0:
                self.store_buffer()
            t += 1
            _, _, rotation = self.get_odom()
            delta_angle = _norm_heading(rotation - pre_angle)
            acc_angle_turn += delta_angle
            pre_angle = rotation
            rospy.loginfo('acc_angle_turn:{}, rotation:{}, goal_angle:{}'.format(
                acc_angle_turn, rotation, goal_angle))
        self.cmd_vel.publish(Twist())
        self.stop_and_sleep()  # stop 1 ms before rotate

    def go_straight(self, goal_distance, backtrack):
        rospy.loginfo("go straight")
        x_start, y_start, rotation = self.get_odom()
        acc_distance = 0
        move_cmd = Twist()
        sign = self.sign if not backtrack else -self.sign

        move_cmd.linear.x = self.linear_speed * sign 
        t = 0
        while goal_distance - acc_distance > self.distance_tolerance and not rospy.is_shutdown():
            self.check_obstacles()  # ! check if there exists obstacles by laserscan in loop
            self.cmd_vel.publish(move_cmd)
            self.rate.sleep()
            if t % self.store_interval == 0:
                self.store_buffer()
            t += 1
            cur_x, cur_y, _ = self.get_odom()
            acc_distance = _calc_distance(
                cur_x - x_start, cur_y - y_start)
            rospy.loginfo('distance:{}, goal_distance:{}, position:{}, {}'.format(
                acc_distance, goal_distance, cur_x, cur_y))
            
            
            if goal_distance - acc_distance < self.slowdown_thresh:
                move_cmd = Twist()
                move_cmd.linear.x = self.slow_speed * sign

        self.cmd_vel.publish(Twist())
        self.stop_and_sleep()

    def get_linear_velocity(self, acc_dist, goal_dist, accelerate_dist=0.06, thred=0.3):
        '''accelerate speed = 0.6, velocity_max = 0.3'''
        if goal_dist > thred:
            if acc_dist <= accelerate_dist:
                #vel = math.sqrt(1.2 * acc_dist)
                vel = 0.3
            elif acc_dist > accelerate_dist and acc_dist <= goal_dist - accelerate_dist:
                vel = 0.3
            else:
                vel = math.sqrt(1.2 * (goal_dist - acc_dist))
        else:
            if acc_dist <= goal_dist / 2:
                #vel = math.sqrt(1.2 * acc_dist)
                vel = 0.3
            else:
                vel = math.sqrt(1.2 * (goal_dist - acc_dist))
        return vel


    def go_curve_line(self, goal_distance, goal_angle, backtrack=False):
        radius = abs((goal_distance / 2) / math.sin(goal_angle / 2))
        move_cmd = Twist()
        sign = self.sign if not backtrack else -self.sign
        move_cmd.linear.x = self.linear_speed * sign 
        move_cmd.angular.z = self.linear_speed / radius if goal_angle >= 0 else -self.linear_speed / radius
        rospy.loginfo(f"goal_angle:{goal_angle}, radius:{radius}, linear:{move_cmd.linear.x}, angular:{move_cmd.angular.z}")
        x_start, y_start, rotation_start = self.get_odom()
        t = 0
        acc_distance = 0
        pre_angle = rotation_start
        acc_angle_turn = 0
        while goal_distance - acc_distance >= self.distance_tolerance and not rospy.is_shutdown():
            self.check_obstacles()  #! check if there exists obstacles by laserscan in loop
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(self.sleep_t)
            if t % self.store_interval == 0:
                self.store_buffer()
            t += 1
            cur_x, cur_y, rotation = self.get_odom()
            acc_distance = _calc_distance(
                cur_x - x_start, cur_y - y_start)
            
            delta_angle = _norm_heading(rotation - pre_angle)
            acc_angle_turn = delta_angle
            
            if goal_distance - acc_distance <= self.slowdown_thresh:
                move_cmd = Twist()
                move_cmd.linear.x = self.slow_speed * sign
                move_cmd.angular.z = self.slow_speed / radius if goal_angle > 0 else -self.slow_speed / radius

        rospy.loginfo('distance:{}, goal_distance:{}, acc_angle_turn:{}, goal_angle:{}'.format(
            acc_distance, goal_distance, acc_angle_turn, goal_angle))
        self.cmd_vel.publish(Twist())
        self.stop_and_sleep()
        
    
    def reach_next_waypoint(self, i, local_rel_pose, backtrack=False):
        ''' 
        Keep publishing Twist msgs, until the internal odometry reaches the goal. 
        return: odom_rel_pose: relative pose in world coordinate
        '''
        dx, dy, dyaw = local_rel_pose
        cur_x, cur_y, cur_yaw = self.get_odom()
        rospy.loginfo('current pose:{}, {}, {}'.format(cur_x, cur_y, cur_yaw))
        self._buffer = []  # reset buffer
        line_angle = atan2(dy, dx) if not backtrack else atan2(-dy, -dx)
        goal_distance = _calc_distance(dx, dy)
        goal_angle = dyaw
        rospy.loginfo("publishing {}th goal: distance:{}, line_angle:{}, goal_angle:{}".format(
            i + 1, goal_distance, line_angle, goal_angle))
        
        odom_rel_pose = None
        while not rospy.is_shutdown():
            '''mode 1: rotate, go straight, and rotate'''
            ## step 1: first rotate to the line between start and goal
            #self.rotate(line_angle)

            ## step 2: go straight
            # self.go_straight(goal_distance, backtrack)

            ## 'step 3: refine the robot's heading to the goal heading
            #self.rotate(goal_angle - line_angle)
            
            '''mode 2: go curve line by controling velocity and angular together'''
            if backtrack:
                self.rotate(goal_angle)
            else:
                self.go_curve_line(goal_distance, goal_angle, backtrack)
                rospy.loginfo("finish reaching {}th waypoint".format(i + 1))

            end_x, end_y, end_yaw = self.get_odom()
            odom_rel_pose = np.array(
                [end_x - cur_x, end_y - cur_x, end_yaw - cur_yaw])
            break
        return odom_rel_pose

    # def get_image(self, msg):
    #     try:
    #         # Convert ROS CompressedImage message to OpenCV2
    #         self.cv2_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    #     except CvBridgeError as e:
    #         rospy.loginfo(e)

    def store_buffer(self):
        img = self.get_real_time_img()
        cur_pose = self.get_odom()
        self._buffer.append((img, cur_pose))

    def get_odom(self):
        try:
            self.tf_listener.waitForTransform(self.odom_frame,
                                              self.base_frame,
                                              rospy.Time(0),
                                              rospy.Duration(self.duration))
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame,
                                                            self.base_frame,
                                                            rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF exception, cannot get_odom()!")
            return
        point = Point(*trans)
        return [point.x, point.y, _norm_heading(_quat_to_angle(*rot))]

    def check_obstacles(self):
        while not self.can_go and not rospy.is_shutdown():
            rospy.sleep(self.sleep_t)

    def get_scan(self, msg):
        self.scan_filter = []
        for i in range(360):
            if i < 45 or i > 135:
                if msg.ranges[i] >= self.scan_error:
                    self.scan_filter.append(msg.ranges[i])

        if min(self.scan_filter) < self.lidar_detect_distance:
            self.cmd_vel.publish(Twist())
            self.cmd_vel.publish(Twist())
            self.can_go = False
            rospy.loginfo("distance to Obstacle {}".format(
                self.lidar_detect_distance))
        else:
            self.can_go = True

    def stop_and_sleep(self, sleep_t=0.5):
        move_cmd = Twist()
        move_cmd.linear.x = 0.05
        self.cmd_vel.publish(move_cmd)
        move_cmd = Twist()
        self.cmd_vel.publish(move_cmd) 
        rospy.sleep(sleep_t)

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.stop_and_sleep()
