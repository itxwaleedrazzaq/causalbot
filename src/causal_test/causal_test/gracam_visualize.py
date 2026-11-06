
#!/home/waleed/anaconda3/envs/tf/bin/python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
from collections import deque
from datetime import datetime
import cv2
import keras
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose, Twist
import tensorflow as tf

from keras_nac.layers import NAC, LAN
from ncps.keras import LTC

from .utils import SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH

keras.config.enable_unsafe_deserialization()

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model_weights",)
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"gradcam_recordings")

SHOW_IMAGE = True
SHOW_GRADCAM = True

GRADCAM_TIMESTEP = -1
GRADCAM_ALPHA = 0.75
GRADCAM_INTERVAL = 5

RECORD_FPS = 5.0

class AiInference(Node):
    def __init__(self):
        super().__init__("ai_inference")

        #loading model
        self.declare_parameter("model", "NAC")
        self.model_name = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("seed", 42)
        self.seed = self.get_parameter("seed").value

        self.declare_parameter("hover_z", 0.5)
        self.hover_z = self.get_parameter("hover_z").value

        self.declare_parameter("stabilization_time", 3.0)
        self.stabilization_time = self.get_parameter("stabilization_time").value

        self.model = keras.models.load_model(os.path.join(WEIGHTS_DIR,f"{self.model_name}.keras"),custom_objects={"NAC": NAC, "LAN": LAN, "LTC": LTC})

        tf.random.set_seed(int(self.seed))
        np.random.seed(int(self.seed))

        with open(f"{WEIGHTS_DIR}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        #Grad-CAM model
        self.conv_layers = []

        for layer in self.model.layers:
            if isinstance(layer,keras.layers.TimeDistributed) and isinstance(layer.layer,keras.layers.Conv2D):
                self.conv_layers.append(layer)

        if not self.conv_layers:
            raise RuntimeError("No TimeDistributed Conv2D layers found for Grad-CAM")

        self.gradcam_model = keras.Model(inputs=self.model.inputs,outputs=[layer.output for layer in self.conv_layers]+[self.model.output])

        self.gradcam_target = 0
        self.frame_count = 0
        self.last_heatmaps = []

        self.frames = deque(maxlen=SEQUENCE_LENGTH)

        self.recording = False
        self.video_writers = []

        os.makedirs(RECORDINGS_DIR,exist_ok=True)

        self.image_sub = self.create_subscription(CompressedImage, '/camera_sensor/image_raw/compressed', self.image_callback, 10)
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_publisher = self.create_publisher(Pose, "/cmd_pos", 10)

        # Initial hover position
        self.hover_pose = Pose()
        self.hover_pose.position.x = 0.0
        self.hover_pose.position.y = 0.0
        self.hover_pose.position.z = float(self.hover_z)
        self.hover_pose.orientation.w = 1.0

        self.stabilized = False
        self.startup_time = self.get_clock().now()

        self.hover_timer = self.create_timer(0.1, self.publish_hover_pose)
        self.velocity_timer = self.create_timer(0.1, self.publish_zero_velocity)

        self.get_logger().info(f"Loaded model: {self.model_name}")
        self.get_logger().info(f"Grad-CAM convolutional layers: {len(self.conv_layers)}")
        self.get_logger().info(f"Hover target: z={self.hover_z:.2f} m")
        self.get_logger().info(f"Stabilization time: {self.stabilization_time:.1f} seconds")
        self.get_logger().info("Press r to start/stop recording, 1 for linear.x, 2 for angular.z, q or Esc to quit")

    # Publish hover position during startup
    def publish_hover_pose(self):
        self.pose_publisher.publish(self.hover_pose)

        elapsed_time = (self.get_clock().now() - self.startup_time).nanoseconds / 1e9

        if elapsed_time >= self.stabilization_time:
            self.stabilized = True
            self.hover_timer.cancel()
            self.get_logger().info("Drone stabilized. AI inference enabled")

    # Publish zero velocity during startup
    def publish_zero_velocity(self):
        if not self.stabilized:
            self.twist_pub.publish(Twist())
        else:
            self.velocity_timer.cancel()

    # Compute Grad-CAM for all convolutional layers
    def compute_gradcams(self,x,target_index=0,timestep=-1):
        x_tensor = tf.convert_to_tensor(x,dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            outputs = self.gradcam_model(x_tensor,training=False)
            prediction = outputs[-1]
            target = prediction[:,target_index]

        heatmaps = []

        for activation in outputs[:-1]:
            gradients = tape.gradient(target,activation)

            if gradients is None:
                heatmaps.append(np.zeros((activation.shape[2],activation.shape[3]),dtype=np.float32))
                continue

            activation_t = activation[:,timestep,:,:,:]
            gradients_t = gradients[:,timestep,:,:,:]

            weights = tf.reduce_mean(gradients_t,axis=(1,2),keepdims=True)

            cam = tf.reduce_sum(weights*activation_t,axis=-1)
            cam = tf.nn.relu(cam)

            cam_min = tf.reduce_min(cam,axis=(1,2),keepdims=True)
            cam_max = tf.reduce_max(cam,axis=(1,2),keepdims=True)

            cam = (cam-cam_min)/(cam_max-cam_min+1e-8)

            heatmaps.append(cam[0].numpy())

        del tape

        return heatmaps

    # Create Grad-CAM overlay
    def make_gradcam_overlay(self,image,heatmap):
        heatmap = cv2.resize(heatmap,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image,1-GRADCAM_ALPHA,heatmap,GRADCAM_ALPHA,0)

        return overlay

    # Add label to image
    def add_label(self,image,text):
        image = image.copy()

        cv2.rectangle(image,(0,0),(image.shape[1],35),(0,0,0),-1)
        cv2.putText(image,text,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

        return image

    # Build Grad-CAM visualization
    def build_gradcam_grid(self,image,heatmaps,linear_x,angular_z):
        images = [self.add_label(image,"Original")]

        for i,heatmap in enumerate(heatmaps):
            overlay = self.make_gradcam_overlay(image,heatmap)
            images.append(self.add_label(overlay,f"Conv2D #{i+1}"))

        rows = []

        for i in range(0,len(images),3):
            row_images = images[i:i+3]

            while len(row_images)<3:
                row_images.append(np.zeros_like(image))

            rows.append(np.hstack(row_images))

        grid = np.vstack(rows)

        cv2.putText(grid,f"linear.x: {linear_x:.3f}",(10,grid.shape[0]-40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(grid,f"angular.z: {angular_z:.3f}",(10,grid.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

        if self.recording:
            cv2.putText(grid,"RECORDING",(grid.shape[1]-180,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)

        return grid

    # Start recording
    def start_recording(self):
        if self.recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.video_writers = []

        names = ["original"]+[f"conv{i+1}" for i in range(len(self.conv_layers))]

        for name in names:
            path = os.path.join(RECORDINGS_DIR,f"{self.model_name}_{name}_{timestamp}.mp4")

            writer = cv2.VideoWriter(path,fourcc,RECORD_FPS,(320,240))

            if not writer.isOpened():
                writer.release()

                for existing_writer in self.video_writers:
                    existing_writer.release()

                self.video_writers = []

                self.get_logger().error(f"Could not create recording: {path}")
                return

            self.video_writers.append(writer)

        self.recording = True

        self.get_logger().info(f"Recording started: {RECORDINGS_DIR}")

    # Stop recording
    def stop_recording(self):
        if not self.recording:
            return

        for writer in self.video_writers:
            writer.release()

        self.video_writers = []
        self.recording = False

        self.get_logger().info(f"Recording saved: {RECORDINGS_DIR}")

    # Save original image and all Grad-CAM overlays
    def record_frames(self,original,heatmaps):
        if not self.recording:
            return

        self.video_writers[0].write(original)

        for i,heatmap in enumerate(heatmaps):
            overlay = self.make_gradcam_overlay(original,heatmap)
            self.video_writers[i+1].write(overlay)

    # Keyboard controls
    def handle_key(self,key):
        if key == ord("1"):
            self.gradcam_target = 0
            self.get_logger().info("Grad-CAM target: linear.x")
            self.last_heatmaps = []

        elif key == ord("2"):
            self.gradcam_target = 1
            self.get_logger().info("Grad-CAM target: angular.z")
            self.last_heatmaps = []

        elif key == ord("r"):
            if not SHOW_GRADCAM:
                self.get_logger().warning("Enable SHOW_GRADCAM to record Grad-CAM")
            elif self.recording:
                self.stop_recording()
            else:
                self.start_recording()

        elif key == ord("q") or key == 27:
            self.stop_recording()
            self.twist_pub.publish(Twist())

            if rclpy.ok():
                rclpy.shutdown()

    # Camera callback
    def image_callback(self,msg):
        try:
            image = cv2.imdecode(np.frombuffer(msg.data,dtype=np.uint8),cv2.IMREAD_COLOR)

            if image is None:
                self.get_logger().warning("Failed to decode image")
                return

            image = cv2.resize(image,(IMAGE_WIDTH,IMAGE_HEIGHT)).astype(np.float32)
            display_frame = cv2.resize(image.astype(np.uint8),(320,240))

            # Drone stabilization
            if not self.stabilized:
                if SHOW_IMAGE or SHOW_GRADCAM:
                    display = display_frame.copy()

                    cv2.putText(display,"Stabilizing drone...",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                    cv2.putText(display,f"Hover z: {self.hover_z:.2f} m",(20,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

                    cv2.imshow(f"{self.model_name} Grad-CAM",display)

                    key = cv2.waitKey(1)&0xFF
                    self.handle_key(key)

                return

            self.frames.append(image)

            if len(self.frames)<SEQUENCE_LENGTH:
                if SHOW_IMAGE or SHOW_GRADCAM:
                    display = display_frame.copy()

                    cv2.putText(display,"Collecting frames...",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

                    cv2.imshow(f"{self.model_name} Grad-CAM",display)

                    key = cv2.waitKey(1)&0xFF
                    self.handle_key(key)

                return

            x = np.expand_dims(np.asarray(self.frames,dtype=np.float32),axis=0)

            # Original AI inference - unchanged
            prediction = self.model.predict(x,verbose=0)[0]

            prediction[0] = np.clip(prediction[0],-0.8,0.8)
            prediction[1] = np.clip(prediction[1],-0.1,0.1)

            # Publish original AI control commands
            twist = Twist()
            twist.linear.x = float(prediction[0])
            twist.angular.z = float(prediction[1])

            self.twist_pub.publish(twist)

            # Grad-CAM visualization only
            if SHOW_GRADCAM:
                self.frame_count += 1

                if self.frame_count % GRADCAM_INTERVAL == 0:
                    self.last_heatmaps = self.compute_gradcams(x,self.gradcam_target,GRADCAM_TIMESTEP)

                if self.last_heatmaps:
                    self.record_frames(display_frame,self.last_heatmaps)

                    grid = self.build_gradcam_grid(display_frame,self.last_heatmaps,prediction[0],prediction[1])

                    cv2.imshow(f"{self.model_name} Grad-CAM",grid)

            elif SHOW_IMAGE:
                display = display_frame.copy()

                cv2.putText(display,f"linear.x: {prediction[0]:.3f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                cv2.putText(display,f"angular.z: {prediction[1]:.3f}",(20,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                cv2.imshow(f"{self.model_name} Predictions",display)

            if SHOW_IMAGE or SHOW_GRADCAM:
                key = cv2.waitKey(1)&0xFF
                self.handle_key(key)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AiInference()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop_recording()

        if rclpy.ok():
            node.twist_pub.publish(Twist())

        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
