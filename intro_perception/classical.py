# """Intentionally weak classical gate detector baseline.

# Gate appearance (as seen facing the gate):
 
# * left post:  black on top, red on the bottom
# * right post: red on top, black on the bottom
# """

# from __future__ import annotations

# import cv2 as cv
# import numpy as np

# from intro_perception.production import prepare_production_namespace

# prepare_production_namespace()

# from perception.tasks.TaskPerceiver import TaskPerceiver
# from perception.tasks.registry import register_perceiver

# from intro_perception.types import GateEstimate


# @register_perceiver(task="gate", algo="intro_classical")
# class ClassicalGatePerceiver(TaskPerceiver):
#     """Find two vertical posts with fixed HSV thresholds."""

#     # ADDED: Default Slider Values
#     def __init__(self):
#         super().__init__(
#             red_h_min=((0, 180), 0),
#             red_h_max=((0, 180), 10),
#             red_s_min=((0, 255), 100),
#             red_v_min=((0, 255), 100),
#             black_v_max=((0, 255), 50),
#         )

#     def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):
#         if frame is None or frame.size == 0:
#             raise ValueError("frame must be a non-empty image")

#         if slider_vals is None:
#             slider_vals = {}

#         # Slider controls for HSV tuning
#         r_h_min = slider_vals.get("red_h_min", 0)
#         r_h_max = slider_vals.get("red_h_max", 10)
#         r_s_min = slider_vals.get("red_s_min", 100)
#         r_v_min = slider_vals.get("red_v_min", 100)
        
#         b_v_max = slider_vals.get("black_v_max", 50)

#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
#         red_mask = cv.inRange(
#             hsv, 
#             np.array((r_h_min, r_s_min, r_v_min)), 
#             np.array((r_h_max, 255, 255))
#         )
        
#         # Broad black mask
#         black_mask = cv.inRange(
#             hsv, 
#             np.array((0, 0, 0)), 
#             np.array((180, 255, b_v_max))
#         )

#         combined_mask = cv.bitwise_or(red_mask, black_mask)
#         combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
#         contours = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

#         candidates = []
#         for contour in contours:
#             area = float(cv.contourArea(contour))
#             x, y, width, height = cv.boundingRect(contour)
#             # Basic vertical aspect ratio check
#             if area >= 30.0 and height >= max(10, int(width * 1.2)):
#                 candidates.append((x, y, width, height, area))
        
#         # Sort by area and pick top 2
#         candidates.sort(key=lambda c: c[-1], reverse=True)
#         selected = sorted(candidates[:2], key=lambda c: c[0])

#         estimate = GateEstimate.invisible()
        
#         if len(selected) == 2:
#             left_post, right_post = selected
            
#             x1 = min(left_post[0], right_post[0])
#             y1 = min(left_post[1], right_post[1])
#             x2 = max(left_post[0] + left_post[2], right_post[0] + right_post[2])
#             y2 = max(left_post[1] + left_post[3], right_post[1] + right_post[3])
 
#             left_center = left_post[0] + left_post[2] / 2.0
#             right_center = right_post[0] + right_post[2] / 2.0
 
#             separation = abs(right_center - left_center) / frame.shape[1]
#             area_fraction = (left_post[-1] + right_post[-1]) / (frame.shape[0] * frame.shape[1])
#             confidence = min(1.0, separation * 1.5 + area_fraction * 8.0)
            
#             estimate = GateEstimate(
#                 True,
#                 confidence,
#                 (left_center + right_center) / (2.0 * frame.shape[1]),
#                 (y1 + y2) / (2.0 * frame.shape[0]),
#                 x1 / frame.shape[1],
#                 y1 / frame.shape[0],
#                 (x2 - x1) / frame.shape[1],
#                 (y2 - y1) / frame.shape[0],
#             ).normalized()
 
#         if not debug:
#             return estimate
            
#         annotated = annotate(frame, estimate)
#         for x, y, width, height, _ in selected:
#             cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
#         return estimate, [annotated, cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)]

# def annotate(frame, estimate, label_prefix="classical"):
#     canvas = frame.copy()
#     if estimate.visible:
#         height, width = frame.shape[:2]
#         x1 = int(estimate.box_x * width)
#         y1 = int(estimate.box_y * height)
#         x2 = int((estimate.box_x + estimate.box_width) * width)
#         y2 = int((estimate.box_y + estimate.box_height) * height)
#         cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
#         label = f"{label_prefix}: gate {estimate.confidence:.2f}"
#         cv.putText(canvas, label, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.65,
#                    (255, 255, 255), 2, cv.LINE_AA)
#     else:
#         cv.putText(canvas, f"{label_prefix}: no gate", (10, 26),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
#     return canvas


from __future__ import annotations

import cv2 as cv
import numpy as np

from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver

from intro_perception.types import GateEstimate

# Tuned on data/generated/train.
MIN_POST_H_FRAC = 0.14    # shortest post, as a fraction of image height
MIN_POST_W_FRAC = 0.02    # thinnest post, as a fraction of image width
MIN_ASPECT = 3.0          # post height / width
MIN_FILL = 0.30           # blob area / bounding-box area
SINGLE_ASPECT = 0.6       # lone post: gate width = post height / this
SINGLE_EDGE_TOL = 0.10    # lone post: the missing post must fall within this fraction of the border
PAIR_PRECISION = (0.52, 0.87, 0.91)  # measured precision with 0 / 1 / 2 posts typed
SINGLE_PRECISION = 0.55
MIN_CONFIDENCE = 0.25     # below this the answer is "no gate"


@register_perceiver(task="gate", algo="intro_classical")
class ClassicalGatePerceiver(TaskPerceiver):
    """Find the left/right gate posts from their colour pattern and pair them."""

    def __init__(self):
        super().__init__(
            red_h_min=((0, 180), 90),     # red hue window start (wraps through 180 -> 0 if min > max)
            red_h_max=((0, 180), 20),     # red hue window end
            red_s_min=((0, 255), 0),      # minimum saturation of a red pixel
            red_v_min=((0, 255), 10),     # minimum brightness of a red pixel
            black_v_max=((0, 255), 120),  # a dark pixel is at most this bright
        )

    def analyze(self, frame, debug, slider_vals=None):
        if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty 3-channel BGR image")

        values = {name: default for name, (_, default) in self.kwargs.items()}
        values.update(slider_vals or {})
        height, width = frame.shape[:2]

        # ---- 1. Ignore the black fill at the border of rotated images ----
        black = (frame.max(axis=2) < 12).astype(np.uint8)
        _, labels = cv.connectedComponents(black)
        edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        border = cv.dilate(np.isin(labels, edge[edge > 0]).astype(np.uint8),
                           np.ones((9, 9), np.uint8))
        valid = border == 0
        if not valid.any():
            valid[:] = True

        # ---- 2. Red mask and dark mask ----
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
        hue, sat, val = cv.split(cv.cvtColor(frame, cv.COLOR_BGR2HSV))
        lightness, a_channel = lab[..., 0], lab[..., 1]

        median = np.median(a_channel[valid])
        sigma = np.median(np.abs(a_channel[valid] - median)) * 1.4826 + 1e-3
        redness = a_channel - median   # how much redder than the water

        low, high = values["red_h_min"], values["red_h_max"]
        in_window = (hue >= low) & (hue <= high) if low <= high else (hue >= low) | (hue <= high)
        red = ((redness > max(9.0, 2.2 * sigma)) & valid & in_window
               & (sat >= values["red_s_min"]) & (val >= values["red_v_min"]))
        dark = ((lightness < 0.6 * cv.GaussianBlur(lightness, (0, 0), 25))
                & (val <= values["black_v_max"]) & valid)
        red = red.astype(np.uint8) * 255
        dark = dark.astype(np.uint8) * 255

        # ---- 3. Find tall thin blobs (the posts) ----
        k = max(3, int(0.035 * height)) | 1
        mask = cv.bitwise_or(red, dark)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN,
                               cv.getStructuringElement(cv.MORPH_RECT, (3, k)))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE,
                               cv.getStructuringElement(cv.MORPH_RECT, (3, 2 * k + 1)))

        posts = []
        for x, y, w, h, area in cv.connectedComponentsWithStats(mask)[2][1:]:
            if (h < MIN_POST_H_FRAC * height or w < MIN_POST_W_FRAC * width
                    or h < MIN_ASPECT * w or area < MIN_FILL * w * h):
                continue
            pixels, excess = mask[y:y + h, x:x + w] > 0, redness[y:y + h, x:x + w]
            third = max(2, h // 3)

            def mean_redness(rows):
                return float(excess[rows][pixels[rows]].mean()) if pixels[rows].sum() > 5 else 0.0

            top, bottom = mean_redness(slice(0, third)), mean_redness(slice(h - third, h))
            # Left post is redder at the bottom, right post is redder at the top.
            kind = "L" if bottom - top > 4 else "R" if top - bottom > 4 else "U"
            posts.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "kind": kind,
                          "top": top, "bottom": bottom, "redness": max(top, bottom)})

        # ---- 4. Pick the best gate: a pair of posts ----
        best = None
        for left in posts:
            for right in posts:
                gap = (right["x"] + right["w"] / 2) - (left["x"] + left["w"] / 2)
                if not 0.35 * width <= gap <= 0.98 * width:
                    continue
                if left["kind"] == "R" or right["kind"] == "L":
                    continue
                if min(left["redness"], right["redness"]) < 3:
                    continue
                ratio = min(left["h"], right["h"]) / max(left["h"], right["h"])
                if ratio < 0.2:
                    continue
                top_gap = abs(left["y"] - right["y"]) / height
                bottom_gap = abs(left["y"] + left["h"] - right["y"] - right["h"]) / height
                quality = ((0.5 + 0.5 * ratio)
                           * (1 - min(top_gap, 0.5)) * (1 - min(bottom_gap, 0.5)))
                typed = (left["kind"] == "L") + (right["kind"] == "R")
                confidence = PAIR_PRECISION[typed] * (0.6 + 0.4 * quality)
                if best is None or confidence > best[0]:
                    x1, x2 = left["x"], right["x"] + right["w"]
                    y2 = max(left["y"] + left["h"], right["y"] + right["h"])
                    y1 = max(min(left["y"], right["y"]), y2 - (x2 - x1))
                    best = (confidence, (x1, y1, x2 - x1, y2 - y1))

        # ---- 5. ...or a single post, only if the other would be outside the frame ----
        for post in posts:
            if post["kind"] == "U" or post["redness"] < 3 or post["h"] < 0.25 * height:
                continue
            gate_width = post["h"] / SINGLE_ASPECT
            if post["kind"] == "R":
                x2 = post["x"] + post["w"]
                if x2 - gate_width > SINGLE_EDGE_TOL * width:
                    continue
                x1 = max(0, x2 - gate_width)
            else:
                x1 = post["x"]
                if x1 + gate_width < (1 - SINGLE_EDGE_TOL) * width:
                    continue
                x2 = min(width, x1 + gate_width)
            strength = min(1.0, abs(post["bottom"] - post["top"]) / 30)
            confidence = (SINGLE_PRECISION * (0.5 + 0.5 * strength)
                          * min(1.0, post["h"] / (0.5 * height)))
            if best is None or confidence > best[0]:
                best = (confidence, (x1, post["y"], x2 - x1, post["h"]))

        # ---- 6. Build the estimate ----
        estimate = GateEstimate.invisible()
        if best is not None and best[0] >= MIN_CONFIDENCE:
            confidence, (x, y, w, h) = best
            estimate = GateEstimate(
                True, confidence,
                (x + w / 2) / width, (y + h / 2) / height,
                x / width, y / height, w / width, h / height,
            ).normalized()

        if not debug:
            return estimate
        masks = cv.merge([dark, np.zeros_like(red), red])  # blue = dark, red = red
        return estimate, [annotate(frame, estimate), masks]


def annotate(frame, estimate, label_prefix="classical"):
    canvas = frame.copy()
    if estimate.visible:
        height, width = frame.shape[:2]
        x1 = int(estimate.box_x * width)
        y1 = int(estimate.box_y * height)
        x2 = int((estimate.box_x + estimate.box_width) * width)
        y2 = int((estimate.box_y + estimate.box_height) * height)
        cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{label_prefix}: gate {estimate.confidence:.2f}"
        cv.putText(canvas, label, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                   (255, 255, 255), 2, cv.LINE_AA)
    else:
        cv.putText(canvas, f"{label_prefix}: no gate", (10, 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    return canvas