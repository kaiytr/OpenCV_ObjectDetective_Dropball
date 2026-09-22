"""웹캠 래퍼. 캡처를 별도 스레드에서 돌린다."""

import threading

import cv2

import config


class Camera:
    """캡처를 스레드로 분리해 게임 루프가 카메라 속도에 묶이지 않게 한다.

    cap.read()는 다음 프레임이 도착할 때까지 30ms 넘게 블로킹한다. 메인 루프에서
    직접 부르면 물리와 렌더까지 30fps로 묶여 공 움직임이 끊겨 보인다. 캡처를
    떼어내면 게임 루프는 60fps로 돌고, 검출만 새 프레임이 올 때 실행하면 된다.
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(config.CAM_INDEX, config.CAM_BACKEND)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"카메라를 열 수 없습니다 (index={config.CAM_INDEX}). "
                "다른 프로그램이 웹캠을 쓰고 있지 않은지 확인하세요."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)

        # 카메라를 연 직후에는 자동 노출이 수렴하느라 화면 밝기가 크게 출렁인다
        # (실측 변동폭 56). 그 구간에서 배경을 찍으면 이후 화면 전체가 물체로
        # 잡히므로, 수렴할 때까지 프레임을 버리며 기다린다.
        ok = False
        frame = None
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError("카메라에서 프레임을 읽지 못했습니다.")

        self._lock = threading.Lock()
        self._frame = self._prepare(frame)
        self._frame_id = 1
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            prepared = self._prepare(frame)
            # 배열을 통째로 교체하므로 읽는 쪽에서 찢어진 프레임을 볼 일이 없다.
            with self._lock:
                self._frame = prepared
                self._frame_id += 1

    @staticmethod
    def _prepare(frame):
        if frame.shape[1] != config.FRAME_W or frame.shape[0] != config.FRAME_H:
            frame = cv2.resize(frame, (config.FRAME_W, config.FRAME_H))
        # 거울 반전이 없으면 손을 오른쪽으로 옮겼을 때 화면에서는 왼쪽으로 가서
        # 조작 자체가 불가능해진다.
        return cv2.flip(frame, 1)

    def read(self):
        """(프레임, 프레임 번호). 새 프레임이 없으면 직전 것을 그대로 돌려준다."""
        with self._lock:
            return self._frame, self._frame_id

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
