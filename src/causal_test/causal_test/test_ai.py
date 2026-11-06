#!/home/waleed/anaconda3/envs/tf/bin/python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
from collections import deque
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
SHOW_IMAGE = True

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

        self.frames = deque(maxlen=SEQUENCE_LENGTH)
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
        self.get_logger().info(f"Hover target: z={self.hover_z:.2f} m")
        self.get_logger().info(f"Stabilization time: {self.stabilization_time:.1f} seconds")

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

    def image_callback(self, msg):
        try:
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warning("Failed to decode image")
                return

            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)

            if not self.stabilized:
                if SHOW_IMAGE:
                    display = cv2.resize(image.astype(np.uint8), (320, 240))
                    cv2.putText(display, "Stabilizing drone...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow(f"{self.model_name} Predictions", display)
                    cv2.waitKey(1)
                return

            self.frames.append(image)
            if len(self.frames) < SEQUENCE_LENGTH:
                return

            x = np.expand_dims(np.asarray(self.frames, dtype=np.float32), axis=0)
            prediction = self.model.predict(x, verbose=0)[0]
            prediction[0] = np.clip(prediction[0], -0.5, 0.5)
            prediction[1] = np.clip(prediction[1], -0.05, 0.05)

            twist = Twist()
            twist.linear.x = float(prediction[0])
            twist.angular.z = float(prediction[1])
            self.twist_pub.publish(twist)

            if SHOW_IMAGE:
                display = cv2.resize(image.astype(np.uint8), (320, 240))
                cv2.putText(display, f"linear.x: {prediction[0]:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"angular.z: {prediction[1]:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(f"{self.model_name} Predictions", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    rclpy.shutdown()

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
        if rclpy.ok():
            node.twist_pub.publish(Twist())
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
