"""Hand-designed Bayesian network v4: 11 ink regions and one hole feature.

Only NumPy is required. No neural network, pretrained feature extractor,
classifier library, automatic structure learning or hidden variable is used.
All image transformations are deterministic functions of the input image.
"""
import numpy as np


MODEL_VERSION = "v4"
GRID_ROWS, GRID_COLS = 4, 3
NORMALIZED_SIZE = 24
CROP_THRESHOLD = 0.1
RELATIVE_DENSITY_CUTS = (0.3, 0.75, 1.25, 1.75)
# Compute all 12 zones first, retaining the full-image mean as denominator.
RASTER_ORDER = (0, 1, 2, 5, 4, 3, 6, 7, 8, 11, 10, 9)
# Retain Z22. The final density region merges bottom-left and bottom-middle.
# Its position denotes the top-left cell of the two-cell region.
GRID_POSITIONS = tuple(divmod(index, GRID_COLS) for index in RASTER_ORDER[:10]) + ((3, 0), None)
FEATURE_NAMES = tuple(f"Z{row+1}{col+1}" for row, col in GRID_POSITIONS[:10]) + ("Z41_Z42", "HoleLocation")
FEATURE_PARENTS = (None,) + tuple(range(10)) + (None,)
FEATURE_STATES = (5,) * 12
HOLE_INK_THRESHOLD = 0.4
HOLE_MIN_AREA = 3
PREPROCESSING = {
    "raw_pixel_range": [0, 255], "normalized_shape": [24, 24],
    "deskew": "Per-image grayscale moment horizontal shear; slope clipped to [-1,1]",
    "crop_threshold_after_dividing_by_255": CROP_THRESHOLD,
    "resize": "Independent height/width bilinear resize with pixel-center coordinates",
    "feature": "10 original zone means plus merged Z41/Z42 mean, each divided by full image mean; plus hole location/count",
    "relative_density_cuts": RELATIVE_DENSITY_CUTS,
    "merged_zone": "Z41_Z42 covers rows 18..23 and columns 0..15 of the normalized image; average raw equal-area densities before quantization",
    "density_denominator": "Mean over all 12 original equal-area zones, including both merged constituents",
    "hole_feature": {
        "foreground_threshold_strictly_greater_than": HOLE_INK_THRESHOLD,
        "background_connectivity": 4, "minimum_hole_area_pixels": HOLE_MIN_AREA,
        "normalized_vertical_centroid": "(mean row + 0.5) / 24",
        "states": ["no hole", "one hole: centroid < 0.4",
                   "one hole: 0.4 <= centroid < 0.6",
                   "one hole: centroid >= 0.6", "at least two holes"],
    },
    "empty_input_image": "All twelve features have state 0",
}


def _validate_X(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError("X must have shape (n_samples, 784)")
    if X.dtype.kind not in "biuf":
        raise ValueError("X must contain numeric pixels")
    if not np.isfinite(X).all() or np.any(X < 0) or np.any(X > 255):
        raise ValueError("X must contain finite pixel values in [0, 255]")
    return X


def _resize_bilinear(image):
    """Resize the cropped grayscale image to 24x24; replicate edge samples."""
    yy = (np.arange(24) + 0.5) * image.shape[0] / 24 - 0.5
    xx = (np.arange(24) + 0.5) * image.shape[1] / 24 - 0.5
    yy = np.clip(yy, 0, image.shape[0] - 1)
    xx = np.clip(xx, 0, image.shape[1] - 1)
    iy, ix = yy.astype(int), xx.astype(int)
    fy, fx = yy - iy, xx - ix
    jy = np.minimum(iy + 1, image.shape[0] - 1)
    jx = np.minimum(ix + 1, image.shape[1] - 1)
    return ((1 - fy[:, None]) * ((1 - fx) * image[iy[:, None], ix]
                                + fx * image[iy[:, None], jx])
            + fy[:, None] * ((1 - fx) * image[jy[:, None], ix]
                            + fx * image[jy[:, None], jx]))


def normalize_images(X):
    """Per-image moment deskew, foreground crop, and 24x24 size normalization.

    With column x and row y, inverse sampling uses x_src=x+k*(y-cy),
    where k=mu_xy/mu_yy. The ratio is computed from this image alone.
    Out-of-image shear samples have intensity zero. Blank/very faint images
    without any post-shear pixel > 0.1 become zero images.
    """
    X = _validate_X(X)
    output = np.zeros((len(X), 24, 24), dtype=np.float32)
    rr, cc = np.indices((28, 28))
    for i, image in enumerate(X.reshape(-1, 28, 28)):
        image = image.astype(np.float64) / 255
        mass = image.sum()
        if mass == 0:
            continue
        cy = (rr * image).sum() / mass
        cx = (cc * image).sum() / mass
        variance = ((rr - cy) ** 2 * image).sum()
        slope = (((rr - cy) * (cc - cx) * image).sum() / variance
                 if variance > 1e-12 else 0.0)
        slope = np.clip(slope, -1, 1)
        source_x = cc + slope * (rr - cy)
        low = np.floor(source_x).astype(int)
        weight = source_x - low
        left = image[rr, np.clip(low, 0, 27)] * ((low >= 0) & (low < 28))
        right = image[rr, np.clip(low + 1, 0, 27)] * ((low + 1 >= 0) & (low + 1 < 28))
        image = left * (1 - weight) + right * weight
        ys, xs = np.nonzero(image > CROP_THRESHOLD)
        if not len(ys):
            continue
        crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        output[i] = _resize_bilinear(crop)
    return output


def _discretize_relative_density(relative):
    # Equality belongs to the higher interval, matching validation experiments.
    return np.searchsorted(RELATIVE_DENSITY_CUTS, relative, side="right")


def _hole_state(ink):
    """Count enclosed 4-connected background components in a binary image."""
    # The outer false guard prevents flat-index neighbor wraparound; the inner
    # free border joins all boundary-connected background into one exterior.
    free = np.pad(~ink, 2, constant_values=True)
    free[[0, -1], :] = False
    free[:, [0, -1]] = False
    width = free.shape[1]
    flat = free.ravel()

    def clear(start):
        flat[start] = False
        stack = [start]
        count = row_sum = 0
        while stack:
            index = stack.pop()
            count += 1
            row_sum += index // width - 2
            for neighbor in (index - width, index + width, index - 1, index + 1):
                if flat[neighbor]:
                    flat[neighbor] = False
                    stack.append(neighbor)
        return count, (row_sum / count + 0.5) / ink.shape[0]

    clear(width + 1)
    holes = []
    for index in np.flatnonzero(flat):
        if flat[index]:
            size, centroid = clear(int(index))
            if size >= HOLE_MIN_AREA:
                holes.append(centroid)
    if not holes:
        return 0
    if len(holes) >= 2:
        return 4
    return 1 + int(holes[0] >= 0.4) + int(holes[0] >= 0.6)


def extract_features(X):
    """Return (n_samples,12) integer states in FEATURE_NAMES order.

    Ten features use original 6x8 regions; the eleventh merges Z41 and Z42
    into a 6x16 region. Merge continuous densities before discretizing.
    The denominator still averages ALL 12 original equal-area zone means.
    The final feature counts/locates enclosed holes in the same normalized
    image. It is a deterministic handcrafted measurement, with no fitting.
    """
    images = normalize_images(X)
    if not len(images):
        return np.empty((0, 12), dtype=np.int64)
    density = images.reshape(-1, 4, 6, 3, 8).mean(axis=(2, 4)).reshape(-1, 12)
    relative = density / np.maximum(density.mean(axis=1, keepdims=True), 1e-6)
    snake = relative[:, RASTER_ORDER]
    merged = (snake[:, 10] + snake[:, 11]) / 2
    zones = _discretize_relative_density(np.column_stack((snake[:, :10], merged)))
    holes = np.asarray([_hole_state(image > HOLE_INK_THRESHOLD) for image in images])
    return np.column_stack((zones, holes)).astype(np.int64)


class BayesNetClassifier:
    """Course protocol: a fixed observed-feature DAG and smoothed frequency CPTs."""

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite, positive number")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = _validate_X(X)
        y = np.asarray(y)
        if y.shape != (len(X),) or not len(X):
            raise ValueError("y must have shape (n_samples,) with n_samples > 0")
        if y.dtype.kind not in "biuf" or not np.isfinite(y).all():
            raise ValueError("y must contain integer labels from 0 to 9")
        if np.any(y != np.floor(y)) or np.any(y < 0) or np.any(y > 9):
            raise ValueError("y must contain integer labels from 0 to 9")
        y = y.astype(np.int64)
        features = extract_features(X)
        self.classes_ = np.arange(10)
        self.feature_names_ = FEATURE_NAMES
        self.feature_parents_ = FEATURE_PARENTS
        self.class_count_ = np.bincount(y, minlength=10)
        self.class_prior_ = (self.class_count_ + self.alpha) / (len(y) + 10 * self.alpha)
        self.class_log_prior_ = np.log(self.class_prior_)
        self.feature_histograms_ = np.array([
            np.bincount(features[:, j], minlength=FEATURE_STATES[j]) for j in range(12)
        ])
        self.cpt_counts_, self.cpt_, self.log_cpt_ = {}, {}, {}
        for j, (name, parent) in enumerate(zip(FEATURE_NAMES, FEATURE_PARENTS)):
            states = FEATURE_STATES[j]
            shape = (10, states) if parent is None else (10, FEATURE_STATES[parent], states)
            counts = np.zeros(shape, dtype=np.int64)
            indices = ((y, features[:, j]) if parent is None else
                       (y, features[:, parent], features[:, j]))
            np.add.at(counts, indices, 1)
            probabilities = (counts + self.alpha) / (
                counts.sum(axis=-1, keepdims=True) + states * self.alpha
            )
            self.cpt_counts_[name] = counts
            self.cpt_[name] = probabilities
            self.log_cpt_[name] = np.log(probabilities)
        return self

    def _feature_log_likelihood(self, features):
        scores = np.broadcast_to(self.class_log_prior_, (len(features), 10)).copy()
        classes = self.classes_[None, :]
        for j, (name, parent) in enumerate(zip(FEATURE_NAMES, self.feature_parents_)):
            state = features[:, j, None]
            table = self.log_cpt_[name]
            scores += (table[classes, state] if parent is None else
                       table[classes, features[:, parent, None], state])
        return scores

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "log_cpt_"):
            raise RuntimeError("Call fit before prediction")
        return self._feature_log_likelihood(extract_features(X))

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Normalized model probabilities; column c corresponds to digit c."""
        scores = self._joint_log_likelihood(X)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)
