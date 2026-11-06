#!/home/waleed/anaconda3/envs/tf/bin/python

import os
import pickle
import select
import sys
import termios
import threading
import time
import tty

import keras
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

HELP = """
-----------------------------------------
 Teleop WASD + Training Data Recorder
-----------------------------------------
   w : forward            s : backward
   a : turn left          d : turn right
   space : stop
   r : start / pause recording
   q / Esc / Ctrl-C : save & quit
-----------------------------------------
"""


class DataRecoder(Node):
    def __init__(self):
        super().__init__("data_recorder")

        # parameters 
        self.declare_parameter("linear_speed", 1.5)    # m/s
        self.declare_parameter("angular_speed", 5.0)   # rad/s
        self.declare_parameter("publish_rate", 10.0)   # Hz
        self.declare_parameter("key_timeout", 0.5) # Reset command time out
        self.declare_parameter("record_fps", 8.0)  # FPS for recording images


        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed = self.get_parameter("angular_speed").value
        publish_rate = self.get_parameter("publish_rate").value
        self.key_timeout = self.get_parameter("key_timeout").value
        self.record_fps = self.get_parameter("record_fps").value

        # recorder settings 
        self.record_interval = 1.0 / self.record_fps
        self.last_record_time = 0.0

        # state 
        self.cmd = Twist()
        self.last_key_time = time.monotonic()
        self.running = True
        self.training_active = False

        self.linear_x = 0.0
        self.angular_z = 0.0
        self.lock = threading.Lock()

        self.images = []
        self.labels = []

        # output files
        DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data",)
        os.makedirs(DATA_DIR, exist_ok=True)
        self.filename = os.path.join(DATA_DIR, f"training_data.pkl",)

        # pub / sub 
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage, "/camera_sensor/image_raw/compressed", self.image_callback, 10)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_cmd)

        # keyboard reader thread 
        self.reader = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.reader.start()
    
        self.get_logger().info(f"Publishing Twist on '/cmd_vel' at {publish_rate:.1f} Hz")
        self.get_logger().info(f"Record FPS: {self.record_fps:.1f}")
        self.get_logger().info(f"Output: {self.filename}")
        self.get_logger().info("Recording is currently PAUSED")

    # Teleop
    def set_cmd(self, linear, angular):
        self.cmd.linear.x = float(linear)
        self.cmd.angular.z = float(angular)
        self.last_key_time = time.monotonic()

    def publish_cmd(self):
        if (time.monotonic() - self.last_key_time) > self.key_timeout:
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
        self.publisher.publish(self.cmd)

    def stop_robot(self):
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        for _ in range(3):
            self.publisher.publish(self.cmd)
            time.sleep(0.05)

    # cmd_vel feedback
    def cmd_vel_callback(self, msg):
        with self.lock:
            self.linear_x = float(msg.linear.x)
            self.angular_z = float(msg.angular.z)

    # Image callback: record + display
    def image_callback(self, msg):
        if not self.running:
            return

        try:
            # CompressedImage -> OpenCV image
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                self.get_logger().warning("Failed to decode image")
                return

            # latest velocity
            with self.lock:
                linear_x = self.linear_x
                angular_z = self.angular_z

            # recording FPS limit
            current_time = time.monotonic()
            should_record = (current_time - self.last_record_time >= self.record_interval)

            if self.training_active and should_record:
                self.images.append(image.copy())
                self.labels.append([linear_x, angular_z])
                self.last_record_time = current_time

            # display
            display_image = image.copy()
            sample_number = len(self.images)

            if self.training_active:
                status_text = "RECORDING: RUNNING"
                status_color = (0, 255, 0)
            else:
                status_text = "RECORDING: PAUSED"
                status_color = (0, 0, 255)

            cv2.rectangle(display_image, (10, 10), (400, 55), (0, 0, 0), -1)
            cv2.putText(display_image, status_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_image, f"Samples: {sample_number}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"Record FPS: {self.record_fps:.1f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"linear.x: {linear_x:.3f}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, f"angular.z: {angular_z:.3f}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, "WASD: drive   SPACE: stop", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2 )
            cv2.putText(display_image, "R: rec/pause   Q/ESC: save & exit", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.imshow("Recording", display_image)

            # has focus are routed into the same handler as the stdin thread.
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_key(chr(key))

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    # Single place for all key handling
    def handle_key(self, key):
        """Returns False when the program should shut down."""
        if key in ("\x03", "\x1b", "q"):            # Ctrl-C, Esc, q
            self.running = False
            return False
        elif key == "w":
            self.set_cmd(self.linear_speed, 0.0)
        elif key == "s":
            self.set_cmd(-self.linear_speed, 0.0)
        elif key == "a":
            self.set_cmd(0.0, self.angular_speed)     # turn left  (CCW)
        elif key == "d":
            self.set_cmd(0.0, -self.angular_speed)    # turn right (CW)
        elif key == " ":
            self.set_cmd(0.0, 0.0)
            self.last_key_time = 0.0                  # force stop
        elif key == "r":
            self.toggle_recording()
        return True

    def toggle_recording(self):
        self.training_active = not self.training_active

        if self.training_active:
            # Reset timer so the first frame is recorded immediately.
            self.last_record_time = time.monotonic() - self.record_interval
            self.get_logger().info("==========================================")
            self.get_logger().info("RECORDING STARTED / RESUMED")
            self.get_logger().info(f"Recording at {self.record_fps:.1f} FPS")
            self.get_logger().info("==========================================")
        else:
            self.get_logger().info("==========================================")
            self.get_logger().info("RECORDING PAUSED")
            self.get_logger().info(f"Current samples: {len(self.images)}")
            self.get_logger().info("Press R to resume")
            self.get_logger().info("==========================================")

    # Keyboard thread (stdin)
    def keyboard_loop(self):
        while self.running and rclpy.ok():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not rlist:
                continue

            key = sys.stdin.read(1)

            if not self.handle_key(key):
                break

    # Save dataset
    def save_data(self):
        if len(self.images) == 0:
            self.get_logger().warning("No images recorded. Nothing to save.")
            return

        if len(self.images) != len(self.labels):
            self.get_logger().error("Image/label count mismatch!")
            return

        data = {"images": self.images, "labels": self.labels}

        self.get_logger().info(f"Saving {len(self.images)} samples...")

        temp_filename = self.filename + ".tmp"

        try:
            with open(temp_filename, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_filename, self.filename)

            file_size = os.path.getsize(self.filename)

            self.get_logger().info("==========================================")
            self.get_logger().info("DATASET SAVED SUCCESSFULLY")
            self.get_logger().info(f"File: {self.filename}")
            self.get_logger().info(f"Samples: {len(self.images)}")
            self.get_logger().info(f"Size: {file_size / (1024 * 1024):.2f} MB")
            self.get_logger().info("==========================================")

        except Exception as e:
            self.get_logger().error(f"FAILED TO SAVE DATASET: {e}")
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass


# Main
def main(args=None):
    rclpy.init(args=args)
    node = DataRecoder()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print(HELP)
    try:
        tty.setraw(fd)                     # read keys one at a time, no echo
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # restore tty
        node.stop_robot()
        node.save_data()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()