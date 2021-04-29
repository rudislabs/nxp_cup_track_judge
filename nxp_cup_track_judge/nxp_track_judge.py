#!/usr/bin/env python3
import os
import sys
import copy
import re
import importlib
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterDescriptor
import sensor_msgs.msg
import nav_msgs.msg
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile
import json
import cv2
if cv2.__version__ < "4.0.0":
    raise ImportError("Requires opencv >= 4.0, "
                      "but found {:s}".format(cv2.__version__))

launch_path = os.path.realpath(__file__).replace("/nxp_track_judge.py","")
track_path = os.path.realpath(os.path.relpath(os.path.join(launch_path,"../tracks")))
output_path = os.path.realpath(os.path.relpath(os.path.join(launch_path,"../judge_output")))

class NXPTrackJudge(Node):

    def __init__(self):

        super().__init__("nxp_track_judge")

        # Get paramaters or defaults
        pyramid_down_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Number of times to pyramid image down.')

        odometry_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Vehicle odometry topic.')

        track_evaluation_image_file_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Input image to evaluate track.')
        
        track_image_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Image output topic for track judge visulization.')

        pixels_per_meter_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Pixels in evaluation image to meters in track.')

        evaluation_frequency_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Frequency to evaluate odometry.')

        namespace_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Namespaceing if needed.')

        
        self.declare_parameter("pyramid_down", 2, 
            pyramid_down_descriptor)

        self.declare_parameter("odometry_topic", "odom", 
            odometry_topic_descriptor)

        self.declare_parameter("evaluation_frequency", 10.0, 
            evaluation_frequency_descriptor)
        
        self.declare_parameter("track_evaluation_image", "InfiniteOverpass.png", 
            track_evaluation_image_file_descriptor)
        
        self.declare_parameter("track_image_topic", "TrackJudge", 
            track_image_topic_descriptor)

        self.declare_parameter("pixels_per_meter", 173.76344086, 
            pixels_per_meter_descriptor)

        self.declare_parameter("namespace", "", 
            namespace_topic_descriptor)


        self.pyrDown = self.get_parameter("pyramid_down").value

        self.trackEvaluationImage = self.get_parameter("track_evaluation_image").value

        self.evaluationFrequency = self.get_parameter("evaluation_frequency").value

        self.pixelsPerMeter = self.get_parameter("pixels_per_meter").value

        self.trackImagePubTopic = self.get_parameter("track_image_topic").value

        self.odometryTopic = self.get_parameter("odometry_topic").value

        self.namespaceTopic = self.get_parameter("namespace").value
  

        # Setup CvBridge
        self.bridge = CvBridge()

        self.outsideBounds = False

        self.newLapFlag = False

        self.endLapFlag = True

        self.currentLapTime = 0

        self.lapStartTime = 0

        self.lapNumber = 0

        self.bestLapTime = None

        self.lapStopTime = 0

        self.outsideBoundsCount = 0

        self.trackPositionOffset = np.array([0.275, 16.5]) # [Y, X] for image coordinates in numpy


        evalImageFullPath = os.path.realpath(os.path.relpath(os.path.join(track_path, self.trackEvaluationImage)))

        self.evalImage = cv2.imread(evalImageFullPath, 0)

        if self.pyrDown > 0:
            for i in range(self.pyrDown):
                self.pixelsPerMeter = self.pixelsPerMeter/2.0
                self.evalImage = cv2.pyrDown(self.evalImage)

        self.trackImageHeight, self.trackImageWidth = self.evalImage.shape[:2]

        self.evalImageBGR = cv2.cvtColor(self.evalImage, cv2.COLOR_GRAY2BGR)

        self.returnedTrackImage = copy.deepcopy(self.evalImageBGR)

        self.lastPositionPixels = np.round(self.trackPositionOffset*self.pixelsPerMeter).astype(int)
        self.lastPositionPixels[0] = self.trackImageHeight-self.lastPositionPixels[0]

        self.resultsJudgeJSON = {}


        if self.namespaceTopic != "":
            self.odometryTopic = '{:s}/{:s}'.format(self.namespaceTopic, self.odometryTopic)

        self.timeEvalStamp = self.get_clock().now().nanoseconds
        
        # Subscribers
        self.odometrySub = self.create_subscription(nav_msgs.msg.Odometry, 
            '/{:s}'.format(self.odometryTopic), 
            self.odometryCallback, qos_profile_sensor_data)

        # Publishers
        self.trackImagePub = self.create_publisher(sensor_msgs.msg.Image,
            '/{:s}'.format(self.trackImagePubTopic), 0)

        self.timeStamp = self.get_clock().now().nanoseconds

        self.writeJSON = True

        if self.writeJSON:

            self.outputJSON = os.path.realpath(os.path.relpath(os.path.join(output_path,'{:s}_results.json'.format(str(self.timeStamp)))))


    def judgeTrack(self, position):

        positionPixels = np.round(np.add(position, self.trackPositionOffset)*self.pixelsPerMeter).astype(int)
        positionPixels[0] = self.trackImageHeight-positionPixels[0] 

        if (self.evalImage[positionPixels[0]][positionPixels[1]] == 0) and not self.outsideBounds:
            self.outsideBounds = True
            self.outsideBoundsCount += 1
            if self.writeJSON:
                self.resultsJudgeJSON['lap_{:d}'.format(self.lapNumber)] = {
                    'OutsideBoundsCount': self.outsideBoundsCount,
                    'CompletionTime': "DNF"
                }
                with open(self.outputJSON, 'w') as judgeJSON:
                    judgeJSON.write(json.dumps(self.resultsJudgeJSON, sort_keys=True, indent=4))

        if (self.evalImage[positionPixels[0]][positionPixels[1]] == 255):
            self.newLapFlag = False
            self.endLapFlag = False
            self.outsideBounds = False            

        if (self.evalImage[positionPixels[0]][positionPixels[1]] == 191) and not self.endLapFlag and not self.newLapFlag:
            self.endLapFlag = True

        if (self.evalImage[positionPixels[0]][positionPixels[1]] == 128) and self.endLapFlag and not self.newLapFlag:
            self.newLapFlag = True
            self.endLapFlag = False
            if self.lapStartTime > 0:
                if self.writeJSON:
                    
                    self.resultsJudgeJSON['lap_{:d}'.format(self.lapNumber)] = {
                        'OutsideBoundsCount': self.outsideBoundsCount,
                        'CompletionTime': float(self.currentLapTime*1e-9)
                    }

                    with open(self.outputJSON, 'w') as judgeJSON:
                        judgeJSON.write(json.dumps(self.resultsJudgeJSON, sort_keys=True, indent=4))

                self.returnedTrackImage = cv2.putText(self.returnedTrackImage, 'Completed Lap {:d} Time: {:.2f} sec'.format(self.lapNumber, float(self.currentLapTime*1e-9)),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
                outputImage = os.path.realpath(os.path.relpath(os.path.join(output_path,'{:s}_lap-{:d}.png'.format(str(self.timeStamp), self.lapNumber))))
                cv2.imwrite(outputImage, self.returnedTrackImage)

                self.lapNumber += 1
                
                self.outsideBoundsCount = 0
                self.lapStopTime = self.timeStamp
                
                if self.bestLapTime is not None:
                    if self.bestLapTime > (self.timeStamp-self.lapStartTime):
                        self.bestLapTime = self.timeStamp-self.lapStartTime
                else:
                    self.bestLapTime = self.timeStamp-self.lapStartTime

                

            self.lapStartTime = self.timeStamp
            self.returnedTrackImage = copy.deepcopy(self.evalImageBGR)


        if self.lapStartTime > 0:
            self.currentLapTime = self.timeStamp-self.lapStartTime
        
        if not np.array_equal(self.lastPositionPixels, positionPixels):
            self.returnedTrackImage = cv2.line(self.returnedTrackImage,
                (self.lastPositionPixels[1], self.lastPositionPixels[0]),
                (positionPixels[1], positionPixels[0]),
                (255,128,128), 4)
        self.returnedTrackImageTime = copy.deepcopy(self.returnedTrackImage)
        self.returnedTrackImageTime = cv2.putText(self.returnedTrackImageTime, 'Lap {:d} Time: {:.2f} sec'.format(self.lapNumber, float(self.currentLapTime*1e-9)),
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        
        if self.bestLapTime is not None:
            self.returnedTrackImageTime = cv2.putText(self.returnedTrackImageTime, 'Best Lap Time: {:.2f} sec'.format(float(self.bestLapTime*1e-9)),
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        
        self.lastPositionPixels = positionPixels
        
        return(self.returnedTrackImageTime)
      
    
    def odometryCallback(self, data):

        self.timeStamp = self.get_clock().now().nanoseconds

        if ((self.timeStamp-self.timeEvalStamp)*1e-9) >= (1.0/float(self.evaluationFrequency)):

            position = np.array([data.pose.pose.position.y, data.pose.pose.position.x])

            self.timeEvalStamp = self.timeStamp

            trackImage = self.judgeTrack(position)

            msg = self.bridge.cv2_to_imgmsg(trackImage, "bgr8")
            msg.header.stamp = data.header.stamp
            self.trackImagePub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = NXPTrackJudge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
