from shapely.geometry import box


class Detection:
    def __init__(self, label, confidence, x_max, x_min, y_max, y_min) -> None:
        self.label = label
        self.confidence = confidence
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.box = box(self.x_min, self.y_min, self.x_max, self.y_max)
