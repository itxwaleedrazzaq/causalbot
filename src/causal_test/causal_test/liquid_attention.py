import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="keras_nac", name="LAN")
class LAN(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        topk: int = 8,
        euler_steps: int = 2,
        use_sink_gate: bool = True,
        activation=None,
        tau_epsilon: float = 1e-6,
        delta_t: float = 0.01,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_pairwise: bool = False,
        return_attention: bool = False,
        return_sequences: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.topk = int(topk)
        self.euler_steps = int(euler_steps)
        self.delta_t = float(delta_t)
        self.use_sink_gate = use_sink_gate
        self.tau_epsilon = tau_epsilon
        self.dropout = dropout
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_pairwise = use_pairwise
        self.return_attention = return_attention
        self.return_sequences = return_sequences

        self.q_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="q_proj")
        self.k_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="k_proj")
        self.v_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="v_proj")
        self.out_dense = tf.keras.layers.Dense(self.d_model, use_bias=self.use_bias, name="out_proj")

        self.gate_in = tf.keras.layers.LSTM(self.d_model, use_bias=use_bias, return_sequences=True, name="gate_in")
        self.gate_out = tf.keras.layers.Dense(1, use_bias=use_bias, name="gate_out")
        self.sink_gate = tf.keras.layers.Dense(self.d_model, activation="sigmoid", use_bias=True, name="gate")

        self.attn_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def split_heads(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        x = tf.reshape(x, (B, T, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        return tf.reshape(x, (B, T, self.d_model))

    def pairwise_concat(self, q, k):
        q_exp = tf.expand_dims(q, axis=3)
        k_exp = tf.expand_dims(k, axis=2)
        q_tile = tf.tile(q_exp, [1, 1, 1, tf.shape(k)[2], 1])
        k_tile = tf.tile(k_exp, [1, 1, tf.shape(q)[2], 1, 1])
        return tf.concat([q_tile, k_tile], axis=-1)

    def sparse_topk_pairwise(self, q, k, K=None):
        if K is None:
            K = self.topk
        scores = tf.einsum("bhqd,bhkd->bhqk", q, k)
        Tk = tf.shape(k)[2]
        K_eff = tf.minimum(K, Tk)
        _, topk_idx = tf.math.top_k(scores, k=K_eff)
        k_gathered = tf.gather(k, topk_idx, batch_dims=2, axis=2)
        q_tiled = tf.tile(tf.expand_dims(q, axis=3), [1, 1, 1, K_eff, 1])
        topk_pairs = tf.concat([q_tiled, k_gathered], axis=-1)
        return topk_pairs, topk_idx

    # ---- exposed for probing: returns the gates AND the retrieved pairs -----
    def gates_and_pairs(self, q, k):
        """Return (phi, tau, topk_idx, pair) for the retrieved (top-K) pairs.

        phi, tau : [B, H, Tq, Tk_ret]  (Tk_ret == topk)
        pair     : [B, H, Tq, Tk_ret, 2*D]  the [q; k] driver u
        """
        if self.use_pairwise:
            pair = self.pairwise_concat(q, k)
            idx = None
        else:
            pair, idx = self.sparse_topk_pairwise(q, k)
        B, H, Tq, Tk = (tf.shape(pair)[0], tf.shape(pair)[1],
                        tf.shape(pair)[2], tf.shape(pair)[3])
        F = tf.shape(pair)[4]
        pair_flat = tf.reshape(pair, (B * H * Tq, Tk, F))
        x = self.gate_in(pair_flat)
        gate_raw = tf.reshape(self.gate_out(x), (-1, H, Tq, Tk))
        phi = tf.nn.relu(gate_raw)
        tau = tf.nn.softplus(gate_raw) + self.tau_epsilon
        return phi, tau, idx, pair

    def call(self, inputs, mask=None, training=None):
        q_in = k_in = v_in = inputs
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            q_in, k_in, v_in = inputs
        elif isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            (q_in, v_in) = inputs
            k_in = q_in

        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
            if mask is None or (hasattr(mask, "shape") and mask.shape == ()):
                mask = None

        q = self.q_dense(q_in)
        k = self.k_dense(k_in)
        v = self.v_dense(v_in)

        qh = self.split_heads(q)
        kh = self.split_heads(k)
        vh = self.split_heads(v)

        phi, tau, topk_idx = self.compute_phi_tau(qh, kh)
        dt = tf.cast(self.delta_t, phi.dtype)
        tau_max = tf.reduce_max(tau)
        dt = tf.minimum(dt, 1.0 / (tau_max + 1e-12))

        a = tf.zeros_like(phi)
        for _ in range(self.euler_steps):
            a = a + dt * (-tau * a + phi)
        attn_logits = a

        if mask is not None:
            mask = tf.cast(mask, attn_logits.dtype)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
            attn_logits = attn_logits + (1.0 - mask) * tf.constant(-1e9, dtype=attn_logits.dtype)

        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        if self.use_pairwise:
            output = tf.matmul(attn_weights, vh)
        else:
            vh_topk = tf.gather(vh, topk_idx, batch_dims=2, axis=2)
            output = tf.reduce_sum(attn_weights[..., tf.newaxis] * vh_topk, axis=3)

        combined = self.combine_heads(output)
        if self.use_sink_gate:
            combined = combined * self.sink_gate(q_in)
        out = self.activation(self.out_dense(combined))

        result = out if self.return_sequences else out[:, -1, :]
        if self.return_attention:
            return result, attn_weights
        return result


    def retrieve_gates(self, x):
        """x: [B, T, F] (KerasTensor or tensor) -> (u_last, phi_last, gamma_last,
        gate_raw_last, idx) where:
            u_last        [B, K, 2D]  the [q;k] driver for the retrieved pairs
            phi_last      [B, K]
            gamma_last    [B, K]      (== tau, same as softplus output)
            gate_raw_last [B, K]      pre-activation a (phi=relu(a), gamma=softplus(a)+eps)
            idx           [B, H, Tq, K]
        Uses the LAST query position (model runs return_sequences=False) and means
        over heads, matching the scalar-per-pair logit of the theory.
        """
        q = self.q_dense(x)
        k = self.k_dense(x)
        qh = self.split_heads(q)
        kh = self.split_heads(k)
        phi, tau, idx, pair = self.gates_and_pairs(qh, kh)
        u_last = tf.reduce_mean(pair[:, :, -1, :, :], axis=1)   # [B,K,2D]
        phi_last = tf.reduce_mean(phi[:, :, -1, :], axis=1)     # [B,K]
        tau_last = tf.reduce_mean(tau[:, :, -1, :], axis=1)     # [B,K]
        gate_raw_last = tf.math.log(
            tf.math.expm1(tf.maximum(tau_last - self.tau_epsilon, 1e-12)))
        return u_last, phi_last, tau_last, gate_raw_last, idx

    def compute_phi_tau(self, q, k):
        if self.use_pairwise:
            pair = self.pairwise_concat(q, k)
            idx = None
        else:
            pair, idx = self.sparse_topk_pairwise(q, k)
        B, H, Tq, Tk, F = (tf.shape(pair)[0], tf.shape(pair)[1], tf.shape(pair)[2],
                           tf.shape(pair)[3], tf.shape(pair)[4])
        pair = tf.reshape(pair, (B * H * Tq, Tk, F))
        x = self.gate_in(pair)
        gate_raw = tf.reshape(self.gate_out(x), (-1, H, Tq, Tk))
        phi = tf.nn.relu(gate_raw)
        tau = tf.nn.softplus(gate_raw) + self.tau_epsilon
        return phi, tau, idx

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "tau_epsilon": self.tau_epsilon,
            "topk": self.topk,
            "euler_steps": self.euler_steps,
            "delta_t": self.delta_t,
            "use_sink_gate": self.use_sink_gate,
            "dropout": self.dropout,
            "activation": tf.keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "use_pairwise": self.use_pairwise,
            "return_attention": self.return_attention,
            "return_sequences": self.return_sequences,
        })
        return cfg


class LanGateExtractor(tf.keras.layers.Layer):

    """Wraps a trained LAN instance and exposes the retrieved-pair driver and the
    two gate heads, so a symbolic KerasTensor can flow through raw-tf ops.

    Outputs (all for the LAST query position, averaged over heads — matching the
    scalar per-pair logit of the paper's theory):
        u         [B, K, 2D]   the [q; k] driver for the retrieved (top-K) pairs
        phi       [B, K]       phi(u)   = relu(gate_raw)
        gamma     [B, K]       gamma(u) = softplus(gate_raw) + eps     (== tau)
        gate_raw  [B, K]       pre-activation a
        idx       [B, H, Tq, K]  retrieved key indices
    """

    def __init__(self, lan, **kwargs):
        super().__init__(**kwargs)
        self.lan = lan
        self.tau_epsilon = lan.tau_epsilon

    def call(self, x, training=None):
        """Returns PER-HEAD quantities for the LAST query position, plus the
        head-averaged view for convenience.

        Shapes (H = num_heads, K = retrieved pairs, D = depth):
            u_heads   [B, H, K, 2D]  the exact [q; k] driver the gate LSTM consumes
            a_heads   [B, H, K]      gate logits (pre-activation), from gate_out
            phi_heads [B, H, K]      relu(a)
            gam_heads [B, H, K]      softplus(a) + eps   (== tau)
            idx       [B, H, Tq, K]
        The model averages these over heads downstream, but the LSTM runs
        per-(head, query), so the faithful driver object is per-head."""
        q = self.lan.q_dense(x)
        k = self.lan.k_dense(x)
        qh = self.lan.split_heads(q)                      # [B,H,Tq,D]
        kh = self.lan.split_heads(k)
        pair, idx = self.lan.sparse_topk_pairwise(qh, kh)  # [B,H,Tq,K,2D]
        B = tf.shape(pair)[0]; H = tf.shape(pair)[1]
        Tq = tf.shape(pair)[2]; K = tf.shape(pair)[3]; F = tf.shape(pair)[4]
        pair_flat = tf.reshape(pair, (B * H * Tq, K, F))
        a_flat = self.lan.gate_out(self.lan.gate_in(pair_flat, training=False),
                                   training=False)        # [B*H*Tq, K, 1]
        a_heads = tf.reshape(a_flat, (B, H, Tq, K))
        a_heads = a_heads[:, :, -1, :]                    # [B,H,K] last query
        phi_heads = tf.nn.relu(a_heads)
        gam_heads = tf.nn.softplus(a_heads) + self.tau_epsilon
        u_heads = pair[:, :, -1, :, :]                    # [B,H,K,2D]
        return u_heads, a_heads, phi_heads, gam_heads, idx

    def get_config(self):
        return super().get_config()
