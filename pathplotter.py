#!/usr/bin/python3
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Quaternion, Pose, Point, PoseStamped
import os
import numpy as np
import cv2
import tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


import tf.transformations

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir,"calib.txt"))
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(3000)
        self.points=[]
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        #ODOMETRY
        self.pose1 = Odometry()
        self.pose1.header.frame_id = "odom"
        self.pose1.child_frame_id = "base_link"
        #PATH
        self.estimated_path = Path()
        self.estimated_path.header.frame_id = "odom"
        #POINTCLOUD
        self.cloud_msg = PointCloud2()
        self.cloud_msg.header.frame_id = "odom"
        self.cloud_msg.height = 1
        self.cloud_msg.fields = [
            pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1)
        ]
        self.cloud_msg.is_bigendian = False
        self.cloud_msg.point_step = 12 
        self.cloud_msg.is_dense = True

    def video_callback(self, data): #CALLBACK FUNCTION FOR GETTING VIDEO
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.process_frame(frame_gray)

    def process_frame(self, frame_gray):
        self.images.append(frame_gray)
        i = len(self.images)-1
        if i > 0:
            q1, q2 = self.get_matches(i)
            if q1 is not None and q2 is not None:
                transf = self.get_pose(q1, q2)
                transf = np.nan_to_num(transf, neginf=0, posinf=0)
                self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(transf))
                euler_angles = (0,yaw(self.cur_pose) - np.pi/2, 0 )
                quaternion = tf.transformations.quaternion_from_euler(*euler_angles)
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "odom"
                pose_msg.pose.position = Point(self.cur_pose[0, 3], self.cur_pose[1,3], self.cur_pose[2,3])
                self.estimated_path.poses.append(pose_msg)
                self.pose1.header.stamp = rospy.Time.now()
                self.pose1.pose.pose.position = pose_msg.pose.position
                self.pose1.pose.pose.orientation = Quaternion(*quaternion)
                self.pose_pub.publish(self.pose1)

                self.path_pub.publish(self.estimated_path)
                points3d_reshaped = np.reshape(self.points3d, (-1, 3))
                transformed_points = np.dot(points3d_reshaped, np.linalg.inv(self.cur_pose[:3, :3].T))
                transformed_points += self.cur_pose[:3, 3]
                if len(self.points)==0:
                    self.points = transformed_points
                else:
                    self.points = np.append(self.points, transformed_points, axis=0)
                self.cloud_msg.header.stamp = rospy.Time.now()
                self.cloud_msg.width = len(self.points)
                self.cloud_msg.row_step = self.cloud_msg.point_step * self.cloud_msg.width
                self.cloud_msg.data = np.asarray(self.points, dtype=np.float32).tobytes()

                self.point_cloud_pub.publish(self.cloud_msg)
                
    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))  #PROJECTION MATRIX
            K = P[0:3, 0:3]     #INTRINSIC MATRIX
        return K, P

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)      #TRANFORMATION MATRIX
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)   #KP AND DESCRIPTORS
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)

        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])   #PIXEL COORDINATES
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])   #PIXEL COORDINATES 2
        return q1, q2

    def get_pose(self, q1, q2):
        E,mask = cv2.findEssentialMat(q1, q2, self.K)      #ESSENTIAL MATRIX

        R, t = self.decomp_essential_mat(E, q1, q2)       #ROTATION AND TRANSLATION

        transformation_matrix = self._form_transf(R, np.squeeze(t))  #TRANSFORMATION MATRIX
        #depth = self.calculate_depth(q1, q2, R, t)
        return transformation_matrix

    def get_3d_points(self, uhom_Q1, uhom_Q2):
        uhom_combined = np.concatenate((uhom_Q1, uhom_Q2), axis=0)
        x_positions = uhom_combined[:, 0]
        y_positions = uhom_combined[:, 1]
        z_positions = uhom_combined[:, 2]
        return [x_positions, y_positions, z_positions]
    
    def decomp_essential_mat(self, E, q1, q2):
        def triangulate_points(R, t):
            T = self._form_transf(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)  #projection matrix

            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)    #HOMOGENOUS 3D COORDINATES OF Q1
            hom_Q2 = np.matmul(T, hom_Q1)

            uhom_Q1 = hom_Q1[:3] / hom_Q1[3, :]     #NON HOMOGENOUS 3D COORDS
            uhom_Q2 = hom_Q2[:3] / hom_Q2[3, :]

            return uhom_Q1.T, uhom_Q2.T

        def sum_z_cal_relative_scale(R, t):
            uhom_Q1, uhom_Q2 = triangulate_points(R, t)
            points3d = uhom_Q1
            sum_of_pos_z_Q1 = sum(uhom_Q1[:, 2] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[:, 2] > 0)

            relative_scale = np.mean(np.linalg.norm(uhom_Q1[:-1] - uhom_Q1[1:], axis=-1) /
                                    np.linalg.norm(uhom_Q2[:-1] - uhom_Q2[1:], axis=-1))

            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale, points3d

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale, self.points3d = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)
        right_pair_idx = np.argmax(z_sums) #max positive z values
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

def yaw(M):
    th = -np.arcsin(M[2, 0]) #calculating yaw
    return th

def main():
    rospy.init_node('visual_odometry')
    data_dir="/home/tanbnz/vo"
    vo = VisualOdometry(data_dir)
    vo.images = []
    vo.cur_pose = np.eye(4)
    vo.estimated_path = Path()
    vo.estimated_path.header.frame_id = "odom"
    vo.pose_pub = rospy.Publisher('/estimated_pose', Odometry, queue_size=10)
    vo.path_pub = rospy.Publisher('/estimated_path', Path, queue_size=10)
    vo.point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
    rospy.Subscriber('/video', Image, vo.video_callback)
    rospy.spin()

if __name__ == "__main__":
    main()
