import numpy as np

EARTH_R = 6371000.0


class LocalProjection:
    """Equirectangular projection around a reference point -> metres.
    Good enough for the city-sized areas in this dataset."""

    def __init__(self, lon0, lat0):
        self.lon0 = lon0
        self.lat0 = lat0
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * EARTH_R

    def __call__(self, lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        x = (lonlat[..., 0] - self.lon0) * self.kx
        y = (lonlat[..., 1] - self.lat0) * self.ky
        return np.stack([x, y], axis=-1)


def project_point_on_segments(p, a, b):
    """Distance from point p to segments a->b, plus fraction t in [0,1] along each."""
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    l2 = np.where(l2 == 0, 1e-12, l2)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / l2, 0.0, 1.0)
    proj = a + ab * t[:, None]
    d = np.linalg.norm(p - proj, axis=1)
    return d, t
