import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
import keras

from .liquid_attention import LAN  

keras.config.enable_unsafe_deserialization()

TF_DTYPE = "float32"
EPS = 1e-6


def load_lan_model(weights_path):
    model = keras.models.load_model(weights_path, custom_objects={"LAN": LAN})
    model.trainable = False
    return model


def find_lan_layer(model):
    for layer in model.layers:
        if layer.__class__.__name__ == "LAN":
            return layer
    # nested (if the LAN sits inside a wrapper)
    for layer in model.layers:
        for sub in getattr(layer, "layers", []):
            if sub.__class__.__name__ == "LAN":
                return sub
    raise RuntimeError("No LAN layer found in the loaded model.")


def build_trunk(model, lan_layer):
    """Return a keras Model mapping the 5-D image sequence -> the LAN layer input
    (i.e. the [B, T, F] driver tensor that feeds the projections)."""
    inp = model.inputs[0]
    target = lan_layer.input  # KerasTensor into the LAN layer
    return keras.Model(inputs=inp, outputs=target)


def build_lan_gates(lan_layer):
    """Return a keras Model mapping a LAN-input tensor [B, T, F]
    -> dict of per-head retrieved-pair driver + gates (last query position).

    Keys:
        u_heads   [B,H,K,2D]   exact [q;k] driver consumed by the gate LSTM
        a_heads   [B,H,K]      gate logits (pre-activation)
        phi_heads [B,H,K]      relu(a)
        gam_heads [B,H,K]      softplus(a)+eps  (== tau)
        idx       [B,H,Tq,K]
        u         [B,K,2D]     head-mean driver  (view only; NOT the LSTM input)
        phi       [B,K]        head-mean phi
        gamma     [B,K]        head-mean gamma
        gate_raw  [B,K]        head-mean a
    """
    from .liquid_attention import LanGateExtractor

    F = lan_layer.input.shape[-1]
    inp = keras.Input(shape=(None, F), dtype=TF_DTYPE, name="lan_in")
    u_h, a_h, phi_h, gam_h, idx = LanGateExtractor(lan_layer, name="gate_extract")(inp)
    return keras.Model(inp, {
        "u_heads": u_h, "a_heads": a_h, "phi_heads": phi_h, "gam_heads": gam_h,
        "idx": idx,
        "u": keras.ops.mean(u_h, axis=1),
        "phi": keras.ops.mean(phi_h, axis=1),
        "gamma": keras.ops.mean(gam_h, axis=1),
        "gate_raw": keras.ops.mean(a_h, axis=1),
    })


# gate-head-only model: raw gate logit for a single pair (for autodiff)
def gate_raw_single_pair(lan_layer, u_pair):
    """DEPRECATED / DEBUG ONLY -- does NOT reproduce the trained model's gate head.

    Presents a single pair as a length-1 LSTM sequence. Because `gate_in` is an
    LSTM run over the *whole* retrieved-pair sequence, the model's gate for pair k
    depends on the LSTM state accumulated over pairs 1..k. This function therefore
    returns the wrong logit (it ignores cross-pair state). Kept only to demonstrate
    that discrepancy. Use `gate_raw_sequence` for the faithful quantity.
    """
    x = tf.reshape(u_pair, (1, 1, u_pair.shape[-1]))
    h = lan_layer.gate_in(x, training=False)
    a = lan_layer.gate_out(h, training=False)
    return tf.squeeze(a)


def gate_raw_heads(lan_layer, u_heads_seq, h_index, k_index):
    """FAITHFUL per-head gate logit for (head h_index, pair k_index).

    u_heads_seq : [H, K, 2D] the per-head retrieved-pair driver sequence, in the
                  exact order the model's gate LSTM consumes it.
    Returns scalar pre-activation a for head h_index, pair k_index.

    This is the object the ODE is applied to. Differentiating it w.r.t. the pair
    gives the true grad_u gamma / grad_u phi of the implemented LAN, because the
    LSTM genuinely runs per (head, query) over this sequence.
    """
    us = tf.convert_to_tensor(u_heads_seq, dtype=TF_DTYPE)     # [H,K,2D]
    x = tf.reshape(us, (us.shape[0], us.shape[1], us.shape[2]))  # [H,K,2D] as batch
    h = lan_layer.gate_in(x, training=False)                    # [H,K,d_model]
    a = lan_layer.gate_out(h, training=False)                   # [H,K,1]
    return a[h_index, k_index, 0]


def gate_raw_sequence_batch(lan_layer, u_seq):
    """Back-compat: all K logits for a single (unbatched) driver sequence."""
    x = tf.reshape(u_seq, (1, u_seq.shape[0], u_seq.shape[-1]))
    h = lan_layer.gate_in(x, training=False)
    a = lan_layer.gate_out(h, training=False)
    return tf.reshape(a, [-1])


def gates_from_logit(a, tau_epsilon=EPS):
    """phi = relu(a); gamma = softplus(a)+eps  (checkpoint-native)."""
    phi = tf.nn.relu(a)
    gamma = tf.nn.softplus(a) + tau_epsilon
    return phi, gamma


# Jacobians (vector-Jacobian products, batched)
def jacobian_rows(fn, x, out_dim):
    """Dense Jacobian d fn(x)/d x  with x:[N] and fn: x -> [out_dim] vector.

    Returns J of shape [out_dim, N] (rows = components of output)."""
    x = tf.convert_to_tensor(x, dtype=TF_DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fn(x)
        y = tf.reshape(y, [-1])
    rows = []
    n = y.shape[0]
    for i in range(n):
        g = tape.gradient(y[i], x)
        rows.append(tf.reshape(g, [-1]))
    return tf.stack(rows)  # [out_dim, N]


def scalar_grad(fn, x):
    """Gradient of scalar fn at x, preserving x's shape. x:[...] -> same shape."""
    x = tf.convert_to_tensor(x, dtype=TF_DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fn(x)
    g = tape.gradient(y, x)
    if g is None:
        raise RuntimeError("Gradient is None: fn does not depend on x.")
    return g


# the coefficient estimator
class LanDCMProbe:
    """Estimates A_out, A, B_int, C_ext for a trained LAN, plus the empirical
    finite-difference check."""

    def __init__(self, weights_path, mode="std"):
        assert mode in ("std", "wp")
        self.mode = mode
        self.model = load_lan_model(weights_path)
        self.lan = find_lan_layer(self.model)
        self.d_model = self.lan.d_model
        self.num_heads = self.lan.num_heads
        self.depth = self.lan.depth
        self.topk = self.lan.topk
        self.trunk = build_trunk(self.model, self.lan)
        self.gate_model = build_lan_gates(self.lan)

    # ---- driver extraction -------------------------------------------------
    def features_at(self, x_images):
        """Images [B,8,75,75,3] -> LAN input features [B,8,F]."""
        return self.trunk.predict(x_images, verbose=0)

    def driver_at(self, x_images):
        """x_images: [B,8,75,75,3] -> dict with u (last position), per retrieved pair."""
        feats = self.features_at(x_images)
        return self.gate_model.predict(feats, verbose=0)

    def driver_from_features(self, feats):
        return self.gate_model.predict(feats, verbose=0)

    # ---- collapsed-level scalars ------------------------------------------
    def scalars_at_driver(self, u_sample):
        """Given a driver *sample* u (shape [2D] or [K,2D]) return the gate scalars.

        In 'std' mode u is the LAN input feature vector Y (F=d_model? no: F is the
        conv feature width). In 'wp' mode u is the [q;k] pair of width 2*d_model.
        We always evaluate the gate head on the pair the LSTM consumes.
        """
        raise NotImplementedError

    # ---- A_out: encoder Jacobian ------------------------------------------
    def encoder_jacobian(self, x_images, driver_index=0, n_probe_dirs=1, seed=0):
        """A_out = J_h(z0) v.

        h maps the *intervention variable* z (a controllable physical scalar, e.g.
        target displacement) to the LAN driver. We do NOT have z, so we expose the
        general Jacobian of the driver w.r.t. a chosen input subspace and let the
        caller project onto a physically meaningful direction v.

        Here we return J = d u / d x  (u = retrie pied-pair driver, flattened)
        evaluated at the given image, restricted to the input pixels via VJP so it
        is tractable (75*75*3*8 = 135000 inputs).
        """
        inp = tf.convert_to_tensor(x_images, dtype=TF_DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(inp)
            g = self.gate_model(inp, training=False)
            u = g["u"]  # [1,K,2D]
        # J rows: d u[driver_index, :] / d x
        u_vec = tf.reshape(u[:, driver_index, :], [-1])
        rows = []
        for i in range(u_vec.shape[0]):
            gi = tape.gradient(u_vec[i], inp)
            rows.append(tf.reshape(gi, [-1]))
        return tf.stack(rows)  # [2D, N_pixels]

    # ---- gate gradients ----------------------------------------------------
    def gate_gradients(self, u_pair, a_out):
        """grad_u gamma(u0) and grad_u phi(u0) (checkpoint-native parametrisation),
        projected onto a_out.

        u_pair : [2D] the pair at the operating point (working-point mode).
        a_out  : [2D] the intervention direction A_out (physically induced).
        Returns (B_int, C_ext) scalars.
        """
        u0 = tf.convert_to_tensor(u_pair, dtype=TF_DTYPE)

        def phi_of_u(u):
            a = gate_raw_single_pair(self.lan, u)
            return tf.nn.relu(a)

        def gamma_of_u(u):
            a = gate_raw_single_pair(self.lan, u)
            return tf.nn.softplus(a) + EPS

        gphi = scalar_grad(phi_of_u, u0)     # [2D]
        ggamma = scalar_grad(gamma_of_u, u0)  # [2D]

        A_out_t = tf.convert_to_tensor(a_out, dtype=TF_DTYPE)
        C_ext = tf.reduce_sum(gphi * A_out_t)
        B_int = -tf.reduce_sum(ggamma * A_out_t)
        return float(B_int), float(C_ext), ggamma.numpy(), gphi.numpy()

    def A_state_coupling(self, u_pair):
        """A = -gamma(u0) = -tau(u0)."""
        a = gate_raw_single_pair(self.lan, tf.convert_to_tensor(u_pair, dtype=TF_DTYPE))
        return float(tf.nn.softplus(a) + EPS)

    # ---- empirical finite-difference check --------------------------------
    def empirical_delta_logit(self, x_images, perturb_dir_img, iota):
        """Perturb the *image* by iota * perturb_dir_img and measure the change in
        the retrieved-pair gate logits (the ODE's a(t)). Returns d(phi - gamma*a)
        via a one-step Euler residual, i.e. the empirical field G(a, u)."""
        x0 = tf.convert_to_tensor(x_images, dtype=TF_DTYPE)
        x1 = x0 + iota * tf.convert_to_tensor(perturb_dir_img, dtype=TF_DTYPE)
        g0 = self.gate_model.predict(x0.numpy(), verbose=0) if hasattr(x0, "numpy") else self.gate_model.predict(np.asarray(x0), verbose=0)
        g1 = self.gate_model.predict(np.asarray(x1), verbose=0)
        # per retrieved pair (last position): field G_hat = phi - gamma * a,
        # with a from the model's own Euler run approximated by its first step.
        def field(g):
            phi = g["phi"]; gam = g["gamma"]; a = g["gate_raw"]
            return phi - gam * a
        return field(g0), field(g1)

