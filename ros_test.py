import rospy
import time
import os
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(10, repeat=True)

print("Starting rospy node initialization test...")
print(f"ROS_IP is: {os.environ.get('ROS_IP')}")
print(f"ROS_HOSTNAME is: {os.environ.get('ROS_HOSTNAME')}")
print(f"ROS_MASTER_URI is: {os.environ.get('ROS_MASTER_URI')}")


start_time = time.time()
print("Calling rospy.init_node()...")

try:
    rospy.init_node('test_node', anonymous=True)
    end_time = time.time()
    print("rospy.init_node() finished.")
    print(f"Initialization took: {end_time - start_time} seconds.")
except rospy.exceptions.ROSInitException as e:
    end_time = time.time()
    print(f"rospy.init_node() failed after {end_time - start_time} seconds.")
    print(f"Error: {e}")

print("Test script finished.")
