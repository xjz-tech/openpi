import os, time, threading, faulthandler, sys, traceback
import rospy

def init_thread():
    print('Thread: calling rospy.init_node()')
    try:
        rospy.init_node('debug_node', anonymous=True)
        print('Thread: init_node finished')
    except Exception as e:
        print('Thread: init_node raised', e)

def dump_after(delay):
    time.sleep(delay)
    print(f'\n==== {delay}s elapsed, dumping all thread stacktraces ====')
    faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
    print('==== end dump ====\n')

print('ROS_MASTER_URI=', os.environ.get('ROS_MASTER_URI'))
print('ROS_IP=', os.environ.get('ROS_IP'))
print('ROS_HOSTNAME=', os.environ.get('ROS_HOSTNAME'))

init_t = threading.Thread(target=init_thread, daemon=True)
init_t.start()

dumper = threading.Thread(target=dump_after, args=(10,), daemon=True)
dumper.start()

init_t.join() 