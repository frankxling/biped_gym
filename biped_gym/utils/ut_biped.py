import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def processObservations(message, agent):
    """
    Helper fuinction to convert a ROS message to joint angles and velocities.
    Check for and handle the case where a message is either malformed
    or contains joint values in an order different from that expected observation_callback
    in hyperparams['jointOrder']
    """
    if not message:
        print("Message is empty")
        return None
    else:
        # # Check if joint values are in the expected order and size.
        # if len(message.name) != len(agent['jointOrder']):
        #     # Check that the message is of same size as the expected message.
        #     if message.name != agent['jointOrder']:
        #         raise Exception

        return np.array(message.position) + np.array(message.velocity)

def positionsMatch(action, lastObservation):
    """
    Compares a given action with the observed position.
    Returns: bool. True if the position is final, False if not.
    """
    acceptedError = 0.01
    for i in range(action.size -1): #lastObservation loses last pose
        if abs(action[i] - lastObservation[i]) > acceptedError:
            return False
    return True
