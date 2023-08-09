# Copyright 2023 Josh Newans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError
import ball_tracker.process_image as proc
from ament_index_python.packages import get_package_share_directory


from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, HailoSchedulingAlgorithm, FormatType)
import cv2
import numpy as np
from multiprocessing import Process, Queue
from PIL import Image as PILImage
import os
import ball_tracker.yolo_detection_postprocess as yolo_post

class HailoInference(Node):

    def __init__(self):
        super().__init__('hailo_inference')
        self.get_logger().info('Starting Hailo inference...')
        
        self.infer_queues_in = {}
        self.infer_queues_out = {}
        self.infer_processes = []
        self.image_sub = self.create_subscription(Image, "/image_in", self.callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_out_pub = self.create_publisher(Image, "/image_out", 1)
        self.image_depth_out_pub = self.create_publisher(Image, "/image_depth_out", 1)
        self.ball_pub  = self.create_publisher(Point,"/detected_ball",1)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.imshow('input', cv_image)
        # cv2.waitKey(1)
        # convert to RGB
        frame_for_infer = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Get height and width from frame
        orig_w = frame_for_infer.shape[1]
        orig_h = frame_for_infer.shape[0]
                
        # Add the preprocessed image to the queues
        for infer_queue_in in self.infer_queues_in.values():
            infer_queue_in.put(frame_for_infer)
        
        # Get the results from the queues
        #----------------------------
        # YOLO network
        #----------------------------
        detections = self.infer_queues_out['yolo'].get()
        cv_image = yolo_post.draw_detections(detections, cv_image, scale_factor_x = orig_w, scale_factor_y = orig_h)
        img_to_pub = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        img_to_pub.header = data.header
        self.image_out_pub.publish(img_to_pub)

        print(detections)
        point_out = Point()
        if detections['num_detections'] != 0:
            for idx in range(detections['num_detections']):
                if detections['classes'][idx] == 1: # person
                    ymin, xmin, ymax, xmax = detections['boxes'][idx]
                    x = (xmin + xmax) / 2 - 0.5 # -0.5 to convert to normalised coordinates
                    y = (ymin + ymax) / 2 - 0.5 # -0.5 to convert to normalised coordinates
                    s = (xmax - xmin)
                    # cast x,y,s to float
                    x = (float) (x)
                    y = (float) (y)
                    s = (float) (s)

                    if (s > point_out.z):                    
                        point_out.x = x
                        point_out.y = y
                        point_out.z = s
                    print(f"Pt {idx}: ({x},{y},{s})")
        if (point_out.z > 0):
            self.ball_pub.publish(point_out)

        depth_frame = self.infer_queues_out['depth'].get()
        img_to_pub = self.bridge.cv2_to_imgmsg(depth_frame, "bgr8")
        img_to_pub.header = data.header
        self.image_depth_out_pub.publish(img_to_pub)


def yolo_inference_worker(infer_queue_in, infer_queue_out, network_group, input_vstreams_params, output_vstreams_params, input_vstream_info):
    
    # Run the inference and post-processing here
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        while True:
            image = infer_queue_in.get()
            if image is None:
                break  # Exit the worker process when None is received
            image_height, image_width, channels = input_vstream_info.shape
            # resize the image to the input resolution of the network
            image_resized = cv2.resize(image, (image_width, image_height))
            infer_results = infer_pipeline.infer(np.expand_dims(image_resized, axis=0).astype(np.uint8))
            # check if the network has NMS on-chip by checking the output_vstreams_params
            # if the output_vstreams_params has '*nms*' key, then the network has NMS on-chip
            for key in output_vstreams_params.keys():
                if 'nms' in key:
                    is_nms = True 
                    detections = yolo_post.post_nms_infer(infer_results, key)
                else:
                    assert False, "The network doesn't have NMS on-chip"    
            infer_queue_out.put(detections)

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    
    x = np.nan_to_num(depth)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    # crop the border due to padding issues in the network
    # mi = np.min(x[2:-2,2:-2])  # get minimum depth
    # ma = np.max(x[2:-2,2:-2])
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    # print(f'{np.min(x)} {np.max(x)}')
    x_ = cv2.applyColorMap(x, cmap)
    return x_

def depth_inference_worker(infer_queue_in, infer_queue_out, network_group, input_vstreams_params, output_vstreams_params, input_vstream_info):
    
    # Run the inference and post-processing here
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        while True:
            image = infer_queue_in.get()
            if image is None:
                break  # Exit the worker process when None is received
            image_height, image_width, channels = input_vstream_info.shape
            # resize the image to the input resolution of the network
            # image_resized = cv2.resize(image, (image_height, image_width))
            image_resized = cv2.resize(image, (image_width,image_height))
            print(image_resized.shape)
            cv2.imshow('input', image_resized)
            infer_results = infer_pipeline.infer(np.expand_dims(image_resized, axis=0).astype(np.uint8))
            # post process
            #logits_dequantized= infer_results[output_vstream_info.name].squeeze()
            logits_dequantized= infer_results['scdepthv3_nyu/conv31'].squeeze()
            logits_dequantized = 1 / (1 + np.exp(-logits_dequantized))
            depth_frame = 1 / (logits_dequantized * 10 + 0.009)
            cropped_depth = depth_frame[2:-2,2:-2] # crop the border due to padding issues in the network
            if (1):
                viz_frame = cropped_depth
            else:
                viz_frame = depth_frame
            viz_frame = visualize_depth(viz_frame)
            cv2.imshow('depth', viz_frame)
            cv2.waitKey(1)
            infer_queue_out.put(viz_frame)

def main(args=None):
    rclpy.init(args=args)
    hailo_inference = HailoInference()
    # Load your HEF files
    # hefs = [HEF(hef_path) for hef_path in args.hefs]
    hef_yolo = HEF(os.path.join(get_package_share_directory('ball_tracker'),'config','yolov5m_wo_spp_60p.hef'))
    hef_depth = HEF(os.path.join(get_package_share_directory('ball_tracker'),'config','scdepthv3_nyu.hef'))
    # Creating the VDevice target with scheduler enabled
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    with VDevice(params) as target:
        infer_processes = []
        # Configure network
        
        #----------------------------
        # YOLO network configuration
        #----------------------------
        yolo_configure_params = ConfigureParams.create_from_hef(hef=hef_yolo, interface=HailoStreamInterface.PCIe)
        yolo_network_groups = target.configure(hef_yolo, yolo_configure_params)
        yolo_network_group = yolo_network_groups[0]

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False.
        yolo_input_vstreams_params = InputVStreamParams.make(yolo_network_group, quantized=True, format_type=FormatType.UINT8)
        yolo_output_vstreams_params = OutputVStreamParams.make(yolo_network_group, quantized=False, format_type=FormatType.FLOAT32)

        # Define dataset params
        yolo_input_vstream_info = hef_yolo.get_input_vstream_infos()[0]
        
        # create infer queues
        yolo_infer_queue_in = Queue(maxsize=1)
        yolo_infer_queue_out = Queue(maxsize=1)
        # append the queues to the queues dict
        hailo_inference.infer_queues_in['yolo'] = yolo_infer_queue_in
        hailo_inference.infer_queues_out['yolo'] = yolo_infer_queue_out
            
        # Create infer process
        yolo_infer_process = Process(target=yolo_inference_worker, 
                                     args=(yolo_infer_queue_in, yolo_infer_queue_out, yolo_network_group, 
                                           yolo_input_vstreams_params, yolo_output_vstreams_params, yolo_input_vstream_info))
        hailo_inference.infer_processes.append(yolo_infer_process)

        #----------------------------
        # SCDepth network configuration
        #----------------------------
        depht_configure_params = ConfigureParams.create_from_hef(hef=hef_depth, interface=HailoStreamInterface.PCIe)
        depht_network_groups = target.configure(hef_depth, depht_configure_params)
        depht_network_group = depht_network_groups[0]
        depht_network_group_params = depht_network_group.create_params()

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False .
        depht_input_vstreams_params = InputVStreamParams.make(depht_network_group, quantized=True, format_type=FormatType.UINT8)
        depht_output_vstreams_params = OutputVStreamParams.make(depht_network_group, quantized=False, format_type=FormatType.FLOAT32)

        # Define dataset params
        depth_input_vstream_info = hef_depth.get_input_vstream_infos()[0]
        depth_output_vstream_info = hef_depth.get_output_vstream_infos()[0]
        depth_image_height, depth_image_width, depth_channels = depth_input_vstream_info.shape
        

        
        # create infer queues
        depth_infer_queue_in = Queue(maxsize=1)
        depth_infer_queue_out = Queue(maxsize=1)
        # append the queues to the queues dict
        hailo_inference.infer_queues_in['depth'] = depth_infer_queue_in
        hailo_inference.infer_queues_out['depth'] = depth_infer_queue_out
        
            
        # Create infer process
        depth_infer_process = Process(target=depth_inference_worker, 
                                        args=(depth_infer_queue_in, depth_infer_queue_out, depht_network_group,
                                        depht_input_vstreams_params, depht_output_vstreams_params, depth_input_vstream_info))        
        hailo_inference.infer_processes.append(depth_infer_process)

        
        # Start the worker processes
        for process in hailo_inference.infer_processes:
            process.start()

        while rclpy.ok():
            rclpy.spin_once(hailo_inference)

        # Once the node is stopped, clean up the worker processes
        for infer_queue in hailo_inference.infer_queues_in.values():
            infer_queue.put(None)  # Tell the worker processes to exit
        for process in infer_processes:
            process.join()  # Wait for the worker processes to exit

        hailo_inference.destroy_node()
        rclpy.shutdown()

