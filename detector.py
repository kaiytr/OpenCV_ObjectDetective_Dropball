"""고정 배경 차분으로 현실 물체의 윤곽 다각형을 추출한다.

적응형 배경 모델(MOG2)을 쓰지 않는 이유: 물체를 가만히 두면 몇 초 만에
배경으로 학습해버려서 콜라이더가 사라진다. 이 게임에서는 물건을 대놓고
공을 여러 번 굴려야 하므로 배경을 고정하고 필요할 때만 재보정한다.
"""

import cv2
import numpy as np

import config


class Detector:
    def __init__(self):
        self.background = None
        # 최적값은 조명과 배경에 따라 달라서 실행 중에 키로 조절할 수 있게
        # 상수를 그대로 쓰지 않고 인스턴스 값으로 들고 있는다.
        self.diff_threshold = config.DIFF_THRESHOLD
        self.min_area = config.MIN_CONTOUR_AREA
        self._accum = None  # 마스크 시간 평균 누적기
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
        )

    @property
    def ready(self):
        return self.background is not None

    def calibrate(self, frame):
        """현재 프레임을 기준 배경으로 저장한다."""
        self.background = self._prepare(frame)
        self._accum = None

    def adjust_threshold(self, delta):
        self.diff_threshold = int(np.clip(self.diff_threshold + delta, 5, 120))

    def adjust_min_area(self, delta):
        self.min_area = float(np.clip(self.min_area + delta, 200, 30000))

    def process(self, frame):
        """(다각형 목록, 디버그용 이진 마스크)를 돌려준다.

        다각형은 각각 (N, 2) int32 배열이며 카메라 픽셀 좌표계를 쓴다.
        """
        if self.background is None:
            return [], np.zeros((config.FRAME_H, config.FRAME_W), dtype=np.uint8)

        gray = self._compensate(self._prepare(frame))
        diff = cv2.absdiff(self.background, gray)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # 열기로 점 노이즈를 지우고, 닫기와 팽창으로 끊긴 덩어리를 잇는다.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=config.DILATE_ITER)
        mask = self._stabilize(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        big.sort(key=cv2.contourArea, reverse=True)

        polys = []
        for contour in big[: config.MAX_CONTOURS]:
            eps = config.APPROX_EPS_RATIO * cv2.arcLength(contour, True)
            poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
            if len(poly) >= 3:
                polys.append(poly.astype(np.int32))
        return polys, mask

    def _compensate(self, gray):
        """화면 전체 밝기가 배경과 어긋난 만큼 되돌린다.

        자동 노출이나 조명 변화로 화면이 통째로 밝아지면 배경과의 차이가 커져
        아무것도 없는데 전 화면이 물체로 잡힌다. 평균 대신 중앙값을 쓰는 이유는
        평균은 화면에 들어온 물체 자체에 끌려가기 때문이다. 물체가 화면 절반을
        넘지 않는 한 중앙값은 배경 쪽에 남는다.
        """
        if not config.BRIGHTNESS_COMPENSATION:
            return gray
        step = config.BRIGHTNESS_SAMPLE_STEP
        sample_now = gray[::step, ::step].astype(np.int16)
        sample_bg = self.background[::step, ::step].astype(np.int16)
        shift = float(np.median(sample_bg - sample_now))
        if abs(shift) < 1.0:
            return gray
        return cv2.add(gray, shift)

    def _stabilize(self, mask):
        """마스크를 시간 방향으로 평균내 한두 프레임 깜빡이는 노이즈를 걸러낸다.

        센서 노이즈는 매 프레임 자리를 옮기며 나타났다 사라지지만 진짜 물체는
        계속 같은 자리에 있다. 몇 프레임 연속으로 잡힌 영역만 통과시킨다.
        대가는 물체 움직임에 대한 반응이 두 프레임 정도 늦어지는 것이다.
        """
        scaled = mask.astype(np.float32) * (1.0 / 255.0)
        if self._accum is None or self._accum.shape != scaled.shape:
            self._accum = scaled
        else:
            cv2.accumulateWeighted(scaled, self._accum, config.MASK_SMOOTHING)
        stable = (self._accum > 0.5).astype(np.uint8) * 255
        return stable

    @staticmethod
    def _prepare(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)
