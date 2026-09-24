"""
2D soft-body model of C. elegans crawling on agar.

The worm lies on its side, so it bends in its dorsal/ventral plane. The body
is a chain of N segments (one per body-wall muscle row). Muscles set a
preferred curvature per segment; the actual curvature relaxes towards it.
Movement follows from resistive force theory: every piece of the body feels
drag that is much larger sideways (normal) than lengthwise (tangential), and
the body moves so that the total force and torque on it are zero
(inertia-free, as for a 1 mm animal in a gel).
"""
import math

import numpy as np

N_SEGMENTS = 24          # = body-wall muscle rows
BODY_LENGTH = 1.0        # mm
DRAG_RATIO = 30.0        # normal / tangential drag on agar (Berri et al. 2009)


class WormBody:
    def __init__(self, n=N_SEGMENTS, length=BODY_LENGTH, drag_ratio=DRAG_RATIO,
                 relax_time=0.12, pos=(0.0, 0.0), heading=0.0):
        self.n = n
        self.ds = length / n
        self.length = length
        self.c_t = 1.0
        self.c_n = drag_ratio
        self.relax_time = relax_time            # s, muscle -> curvature lag
        self.kappa = np.zeros(n - 1)            # curvature at the n-1 inner joints [1/mm]
        self.center = np.array(pos, dtype=float)
        self.angle = heading                    # body orientation (mean segment angle)

    # ------------------------------------------------------------ geometry --
    def _shape(self, kappa):
        """Node positions in the body frame: centroid at 0, mean angle 0.
        Node 0 is the head."""
        theta = np.concatenate([[0.0], np.cumsum(kappa * self.ds)])
        # the head points along +x, so the body extends backwards (-x)
        theta = theta - theta.mean()
        seg = -self.ds * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        nodes = np.vstack([[0.0, 0.0], np.cumsum(seg, axis=0)])
        return nodes - nodes.mean(axis=0)

    def _to_world(self, local, center, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + center

    def nodes(self):
        """World positions of the n+1 nodes, head first."""
        return self._to_world(self._shape(self.kappa), self.center, self.angle)

    @property
    def head(self):
        return self.nodes()[0]

    @property
    def heading(self):
        p = self.nodes()
        d = p[0] - p[1]
        return math.atan2(d[1], d[0])

    # ------------------------------------------------------------ physics --
    def step(self, target_kappa, dt):
        """Advance by dt seconds towards the muscle-defined curvature."""
        k_new = self.kappa + (np.asarray(target_kappa) - self.kappa) * min(1.0, dt / self.relax_time)
        p0 = self._to_world(self._shape(self.kappa), self.center, self.angle)
        p1 = self._to_world(self._shape(k_new), self.center, self.angle)
        v_shape = (p1 - p0) / dt

        # unit tangents at nodes (central differences), normals rotated 90 deg
        t = np.gradient(p0, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
        w = np.full(len(p0), self.ds)
        w[[0, -1]] *= 0.5                                   # trapezoid weights
        r = p0 - self.center

        # drag tensor per node: K = c_t t t^T + c_n n n^T
        K = (self.c_t * t[:, :, None] * t[:, None, :]
             + self.c_n * nrm[:, :, None] * nrm[:, None, :]) * w[:, None, None]
        # velocity of node i = v_shape + U + Omega * perp(r_i), perp(r) = (-r_y, r_x)
        perp = np.stack([-r[:, 1], r[:, 0]], axis=1)
        # force f_i = -K_i v_i ; solve sum f = 0 and sum r x f = 0 for (U, Omega)
        A = np.zeros((3, 3))
        b = np.zeros(3)
        KU = K                                   # d f_i / d U   = -K_i
        KO = np.einsum("nij,nj->ni", K, perp)    # d f_i / d Omega = -K_i perp_i
        Ks = np.einsum("nij,nj->ni", K, v_shape)
        A[0:2, 0:2] = KU.sum(axis=0)
        A[0:2, 2] = KO.sum(axis=0)
        b[0:2] = -Ks.sum(axis=0)
        cross = lambda a: r[:, 0] * a[..., 1] - r[:, 1] * a[..., 0]
        A[2, 0] = cross(KU[:, :, 0]).sum()
        A[2, 1] = cross(KU[:, :, 1]).sum()
        A[2, 2] = cross(KO).sum()
        b[2] = -cross(Ks).sum()
        U0, U1, omega = np.linalg.solve(A, b)

        self.center = self.center + np.array([U0, U1]) * dt
        self.angle += omega * dt
        self.kappa = k_new

    def curvature_at_rows(self):
        """Curvature per muscle row (n values), for proprioception."""
        k = np.concatenate([[self.kappa[0]], self.kappa, [self.kappa[-1]]])
        return 0.5 * (k[:-1] + k[1:])


def travelling_wave(n, t, freq=0.5, wavelength=0.65, amp=6.0, direction=1):
    """Reference curvature wave, head -> tail for direction=+1 (forward)."""
    s = (np.arange(n - 1) + 1) / n
    return amp * np.sin(2 * math.pi * (s / wavelength - direction * freq * t))


if __name__ == "__main__":
    # sanity check: a head->tail wave must move the worm forwards (head first)
    for direction in (1, -1):
        body = WormBody()
        dt = 0.01
        for k in range(int(10 / dt)):
            body.step(travelling_wave(body.n, k * dt, direction=direction), dt)
        disp = body.center
        head_dir = np.array([math.cos(body.heading), math.sin(body.heading)])
        print(f"wave direction {direction:+d}: moved {np.linalg.norm(disp):.2f} mm in 10 s, "
              f"along heading {np.dot(disp, head_dir):+.2f} mm "
              f"(speed {np.linalg.norm(disp) / 10 * 1000:.0f} um/s)")
