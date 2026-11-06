import csv
import os
import pickle
import time
from collections import deque

import cv2
import keras
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

# The layer must be importable so Keras can resolve "keras_nac>LAN" when the
# checkpoint is deserialised. We reuse the transcription used by the probe.
from .liquid_attention import LAN, LanGateExtractor  # noqa: F401
from .dcm_coefficients import (
    EPS,
    find_lan_layer,
)
from .utils import (
    SEQUENCE_LENGTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)

keras.config.enable_unsafe_deserialization()

try:
    from ament_index_python.packages import get_package_share_directory
    _PKG_SHARE = get_package_share_directory("causal_test")
    DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "model_weights")
except Exception:
    DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "model_weights")


class AiInferenceCoeff(Node):
    def __init__(self):
        super().__init__("ai_inference_coeff")

        # ---------------- parameters -----------------------------------
        self.declare_parameter("model", "LAN")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("coeff_every", 5)
        self.declare_parameter("A_out_seed", 0)
        self.declare_parameter("iota", 1e-3)
        self.declare_parameter("weights_dir", DEFAULT_WEIGHTS_DIR)
        self.declare_parameter("show_image", True)

        self.model_name = self.get_parameter("model").get_parameter_value().string_value
        self.coeff_every = int(self.get_parameter("coeff_every").value)
        self.A_out_seed = int(self.get_parameter("A_out_seed").value)
        self.iota = float(self.get_parameter("iota").value)
        weights_dir = self.get_parameter("weights_dir").get_parameter_value().string_value
        self.show_image = bool(self.get_parameter("show_image").value)

        csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"trajectories")
        self.csv_path = os.path.join(csv_dir,f"coefficienct_trajectory.csv")
        

        # ---------------- model + scaler -------------------------------
        model_file = os.path.join(weights_dir, f"{self.model_name}.keras")
        self.model = keras.models.load_model(
            model_file, custom_objects={"LAN": LAN, "LanGateExtractor": LanGateExtractor})
        with open(os.path.join(weights_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        self.get_logger().info(f"Loaded model: {self.model_name} from {model_file}")

        # ---------------- coefficient machinery ------------------------
        self.lan = None
        self.trunk = None
        self.gate_model = None
        self.A_out = None
        self.has_coeffs = False
        try:
            self.lan = find_lan_layer(self.model)
            self._build_coeff_graphs()
            rng = np.random.default_rng(self.A_out_seed)
            d2 = self.lan.depth * 2
            v = rng.standard_normal(d2).astype("float32")
            self.A_out = v / np.linalg.norm(v)
            self.has_coeffs = True
            self.get_logger().info(
                f"Coefficient probe ENABLED (LAN d_model={self.lan.d_model}, "
                f"heads={self.lan.num_heads}, topk={self.lan.topk})")
        except Exception as e:  # non-LAN models: keep driving, skip coefficients
            self.get_logger().warn(
                f"Coefficient probe DISABLED for model '{self.model_name}': {e}")


        n_aout = len(self.A_out) if self.A_out is not None else 0
        self._aout_cols = [f"A_out_{i}" for i in range(n_aout)]

        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "t_wall", "t_ros", "step", "linear_x", "angular_z",
            "A", "B_int", "C_ext", "a0", "phi", "gamma",
            "dG_pred_partial", "dG_emp", "omitted_term", "K", "H",
        ] + self._aout_cols)
        self._csv_file.flush()
        self.get_logger().info(f"Writing coefficients to {self.csv_path}")
        self._step = 0

        # ---------------- ROS plumbing --------------------------------
        self.frames = deque(maxlen=SEQUENCE_LENGTH)
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera_sensor/image_raw/compressed",
            self.image_callback, 10)
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    # coefficient graph builders (mirror lan_dcm_probe/dcm_coefficients.py)
    def _build_coeff_graphs(self):
        import tensorflow as tf
        import keras

        lan = self.lan
        # trunk: full model input -> LAN input features
        self.trunk = keras.Model(inputs=self.model.inputs[0], outputs=lan.input)

        # per-head retrieved-pair driver + gates, for the last query position
        F = lan.input.shape[-1]
        inp = keras.Input(shape=(None, F), dtype="float32", name="lan_in")
        u_h, a_h, phi_h, gam_h, idx = LanGateExtractor(lan, name="gate_extract")(inp)
        self.gate_model = keras.Model(inp, {
            "u_heads": u_h, "a_heads": a_h, "phi_heads": phi_h,
            "gam_heads": gam_h, "idx": idx,
            "u": keras.ops.mean(u_h, axis=1),
            "gate_raw": keras.ops.mean(a_h, axis=1),
        })
        self._tf = tf

    def _gate_logits(self, us):
        """All per-head logits a for a driver sequence [H,K,2D] -> [H,K]."""
        tf = self._tf
        h = self.lan.gate_in(us, training=False)          # [H,K,d_model]
        a = self.lan.gate_out(h, training=False)          # [H,K,1]
        return tf.reshape(a, (tf.shape(us)[0], tf.shape(us)[1]))

    def _all_coefficients(self, u_heads):
        """Per-(head,pair) coefficients via a single batched Jacobian.

        All H*K logits come from one LSTM pass over the [H,K,2D] driver. We then
        take ONE tape.batch_jacobian over the four [M]-valued fields
        (a, gamma, phi, G), giving d q_m / d u for every pair m in a single
        backward pass. For each m we keep only pair m's own [2D] slot.

        Cost note: the LSTM backward over M=H*K batch copies dominates (~2 s on
        CPU for d_model=64/16 heads/8 pairs). Throttle with `coeff_every`.
        """
        tf = self._tf
        us = tf.convert_to_tensor(u_heads, dtype="float32")      # [H,K,2D]
        H = int(us.shape[0]); K = int(us.shape[1]); D2 = int(us.shape[2])
        M = H * K
        a_out = np.asarray(self.A_out, dtype="float32")          # [2D] numpy

        pairs = [(m // K, m % K) for m in range(M)]

        def jac_of_stacked(fns):
            """One batched Jacobian for several [M]-valued fields at once.

            fns: list of callables us[H,K,2D] -> [M]. We concatenate their
            outputs to [P*M] and take a single batch_jacobian, so all four
            fields cost one backward pass instead of four."""
            us_flat = tf.reshape(us, (1, M * D2))
            P = len(fns)

            def f(x):
                usr = tf.reshape(x, (H, K, D2))
                return tf.reshape(tf.concat([fn(usr) for fn in fns], axis=0), (1, P * M))

            with tf.GradientTape() as jt:
                jt.watch(us_flat)
                out = f(us_flat)
            J = jt.batch_jacobian(out, us_flat)                  # [1, P*M, M*D2]
            J = tf.reshape(J, (P, M, H, K, D2)).numpy()
            return [np.stack([J[p, m][pairs[m]] for m in range(M)]) for p in range(P)]

        def f_a(u):
            return tf.reshape(self._gate_logits(u), (M,))

        def f_g(u):
            return tf.reshape(tf.nn.softplus(self._gate_logits(u)) + EPS, (M,))

        def f_p(u):
            return tf.reshape(tf.nn.relu(self._gate_logits(u)), (M,))

        def f_G(u):
            a = self._gate_logits(u)
            return tf.reshape(-(tf.nn.softplus(a) + EPS) * a + tf.nn.relu(a), (M,))

        ga, gg, gp, gf = jac_of_stacked([f_a, f_g, f_p, f_G])

        with tf.GradientTape() as vt:
            vt.watch(us)
            a = self._gate_logits(us)
            gam = tf.nn.softplus(a) + EPS
            phi = tf.nn.relu(a)
        a_v = a.numpy().reshape(M)
        gam_v = gam.numpy().reshape(M)
        phi_v = phi.numpy().reshape(M)

        # [M,D2] @ [D2] -> [M]   (numpy throughout)
        C_ext = gp.dot(a_out)
        B_int = -(gg.dot(a_out))
        partial = B_int * a_v + C_ext
        total = gf.dot(a_out)

        return dict(
            A=-gam_v, B_int=B_int, C_ext=C_ext,
            a0=a_v, phi=phi_v, gamma=gam_v,
            partial=partial, total=total,
        )


    def _empirical_dG(self, u_heads):
        """Empirical dG/diota per (h,k): perturb EVERY retrieved pair along A_out
        and read each pair's own field change in one forward pass."""
        tf = self._tf
        us = tf.convert_to_tensor(u_heads, dtype="float32")      # [H,K,2D]
        H, K, D2 = int(us.shape[0]), int(us.shape[1]), int(us.shape[2])
        a_out = tf.convert_to_tensor(self.A_out, dtype="float32")
        delta = tf.broadcast_to(a_out[None, None, :], (H, K, D2))

        def field(usq):
            a = self._gate_logits(usq)
            return -(tf.nn.softplus(a) + EPS) * a + tf.nn.relu(a)

        G0 = field(us)
        G1 = field(us + self.iota * delta)
        return ((G1 - G0) / self.iota).numpy()                   # [H,K]

    def _coefficients_for(self, feats):
        """Average the coefficients over all heads and retrieved pairs."""
        g = self.gate_model.predict(feats, verbose=0)
        u_heads = g["u_heads"][0]           # [H, K, 2D]
        H, K = int(u_heads.shape[0]), int(u_heads.shape[1])

        d = self._all_coefficients(u_heads)
        d_emp = self._empirical_dG(u_heads)

        agg = {key: float(np.mean(d[key]))
               for key in ("A", "B_int", "C_ext", "a0", "phi", "gamma",
                           "partial", "total")}
        agg["dG_emp"] = float(np.mean(d_emp))
        agg["omitted_term"] = agg["total"] - agg["partial"]
        agg["K"], agg["H"] = int(K), int(H)
        return agg

    # ------------------------------------------------------------------
    # ROS callback -- identical control path to test_ai.py, plus logging
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warning("Failed to decode image")
                return
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
            self.frames.append(image)
            if len(self.frames) < SEQUENCE_LENGTH:
                return

            x = np.expand_dims(np.asarray(self.frames, dtype=np.float32), axis=0)

            # ---- control (same as test_ai.py) -------------------------
            prediction = self.scaler.inverse_transform(
                self.model.predict(x, verbose=0))[0]
            prediction[0] = np.clip(prediction[0], -0.8, 0.8)
            prediction[1] = np.clip(prediction[1], -1.5, 1.5)
            twist = Twist()
            twist.linear.x = float(prediction[0])
            twist.angular.z = float(prediction[1])
            self.twist_pub.publish(twist)

            # ---- coefficients + CSV ----------------------------------
            row = None
            if self.has_coeffs and (self._step % self.coeff_every == 0):
                feats = self.trunk.predict(x, verbose=0)
                c = self._coefficients_for(feats)
                row = [
                    f"{time.time():.6f}",
                    f"{self.get_clock().now().nanoseconds / 1e9:.6f}",
                    self._step,
                    f"{twist.linear.x:.6f}", f"{twist.angular.z:.6f}",
                    f"{c['A']:.8g}", f"{c['B_int']:.8g}", f"{c['C_ext']:.8g}",
                    f"{c['a0']:.8g}", f"{c['phi']:.8g}", f"{c['gamma']:.8g}",
                    f"{c['partial']:.8g}", f"{c['dG_emp']:.8g}",
                    f"{c['omitted_term']:.8g}", c["K"], c["H"],
                ]
                # append A_out components (constant per run; written on every row)
                row.extend(f"{float(w):.8g}" for w in self.A_out)
                self.log_coefficients(row)
                self._csv_file.flush()   # keep the CSV complete on a hard stop

            self._step += 1

            if self.show_image:
                display = cv2.resize(image.astype(np.uint8), (320, 240))
                cv2.putText(display, f"linear.x: {twist.linear.x:.3f}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"angular.z: {twist.angular.z:.3f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(f"{self.model_name} Predictions", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

    def log_coefficients(self, row):
        # CSV first; optionally also a throttled ROS log line.
        try:
            self._csv.writerow(row)
        except Exception as e:
            self.get_logger().error(f"CSV write failed: {e}")

    def destroy_node(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
            self.get_logger().info(f"Closed CSV: {self.csv_path}")
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceCoeff()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
