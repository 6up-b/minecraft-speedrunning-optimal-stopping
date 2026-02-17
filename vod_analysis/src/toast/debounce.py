class Debouncer:
    def __init__(self, stable_frames=6, min_conf=0.5):
        self.stable_frames = stable_frames
        self.min_conf = min_conf
        self.prev = None
        self.count = 0

    def update(self, text, conf):
        if text is None or conf < self.min_conf:
            self.prev = None
            self.count = 0
            return None

        if text == self.prev:
            self.count += 1
        else:
            self.prev = text
            self.count = 1

        if self.count == self.stable_frames:
            return {"text": text, "conf": conf}
        return None
