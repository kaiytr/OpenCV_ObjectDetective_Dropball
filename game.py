"""드롭볼 게임 진입점.

공은 오른쪽 위에서 떨어지고 바구니는 왼쪽 아래에 있다. 그냥 두면 공은
오른쪽 바닥으로 빠지므로, 웹캠 앞에 실제 물건을 비스듬히 대어 공을
왼쪽으로 튕겨 보내야 한다.
"""

import sys

import pygame

import config
from camera import Camera
from detector import Detector
from render import Renderer
from world import World

CALIBRATING = "CALIBRATING"
PLAYING = "PLAYING"
RESULT = "RESULT"


class Game:
    def __init__(self):
        pygame.init()
        self.camera = Camera()
        self.detector = Detector()
        self.renderer = Renderer()
        self.world = World()
        self.clock = pygame.time.Clock()

        self.success = 0
        self.fail = 0
        self.debug = False
        self.running = True

        self.polys = []
        self.mask = None
        self.frame_count = 0

        self.state = CALIBRATING
        self.timer = config.CALIBRATION_COUNTDOWN
        self.last_result = None

        # 연출용 타이머. 게임 판정에는 관여하지 않는다.
        self.last_frame_id = -1
        self.new_frame = True
        self.flash = None   # (색, 남은 시간)
        self.pulse = 0.0    # 성공 직후 바구니 발광

    # ------------------------------------------------------------------ 루프
    def run(self):
        try:
            while self.running:
                # 프레임이 급락해도 물리가 폭주하지 않도록 dt에 상한을 둔다.
                dt = min(self.clock.tick(config.TARGET_FPS) / 1000.0, 1 / 30)
                frame, frame_id = self.camera.read()
                if frame is None:
                    continue
                # 캡처가 스레드로 분리돼 게임 루프는 카메라보다 빨리 돈다.
                # 같은 프레임을 다시 검출하는 건 낭비이므로 번호로 걸러낸다.
                self.new_frame = frame_id != self.last_frame_id
                self.last_frame_id = frame_id
                self._handle_events()
                self._update(frame, dt)
                self._draw(frame)
                pygame.display.flip()
        finally:
            self.camera.release()
            pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_b:
                    self._start_calibration()
                elif event.key == pygame.K_d:
                    self.debug = not self.debug
                elif event.key == pygame.K_r:
                    self.success = self.fail = 0
                # 노이즈 최적값은 조명과 배경에 따라 달라 미리 정할 수 없다.
                # 디버그 마스크를 보면서 바로 조절할 수 있게 키를 열어둔다.
                elif event.key == pygame.K_LEFTBRACKET:
                    self.detector.adjust_threshold(-2)
                    self.debug = True
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.detector.adjust_threshold(+2)
                    self.debug = True
                elif event.key == pygame.K_COMMA:
                    self.detector.adjust_min_area(-250)
                    self.debug = True
                elif event.key == pygame.K_PERIOD:
                    self.detector.adjust_min_area(+250)
                    self.debug = True

    def _start_calibration(self):
        self.state = CALIBRATING
        self.timer = config.CALIBRATION_COUNTDOWN
        self.polys = []
        self.world.sync_obstacles([])

    # ------------------------------------------------------------------ 갱신
    def _update(self, frame, dt):
        self.frame_count += 1
        self._update_effects(dt)

        if self.state == CALIBRATING:
            self.timer -= dt
            if self.timer <= 0:
                self.detector.calibrate(frame)
                self._respawn()
                self.state = PLAYING
            return

        # 새 카메라 프레임이 왔을 때만 검출한다. 물리와 렌더는 그보다 자주 돈다.
        if self.new_frame and self.frame_count % config.DETECT_EVERY_N_FRAMES == 0:
            self.polys, self.mask = self.detector.process(frame)
            self.world.sync_obstacles(self.polys)

        if self.state == PLAYING:
            self.world.step(dt)
            if self.world.ball_in_basket():
                self.success += 1
                self._show_result("성공!", config.COLOR_SUCCESS)
            elif self.world.ball_lost():
                self.fail += 1
                self._show_result("실패", config.COLOR_FAIL)

        elif self.state == RESULT:
            self.timer -= dt
            if self.timer <= 0:
                self._respawn()
                self.state = PLAYING

    def _update_effects(self, dt):
        if self.flash:
            color, remaining = self.flash
            remaining -= dt
            self.flash = (color, remaining) if remaining > 0 else None
        if self.pulse > 0:
            self.pulse = max(0.0, self.pulse - dt)

    def _respawn(self):
        self.world.spawn_ball()
        # 잔상을 끊지 않으면 사라진 공과 새 공이 선으로 이어져 보인다.
        self.renderer.reset_ball_trail()

    def _show_result(self, text, color):
        self.state = RESULT
        self.timer = config.RESULT_DISPLAY_SEC
        self.last_result = (text, color)
        self.flash = (color, config.FLASH_SEC)
        if color == config.COLOR_SUCCESS:
            self.pulse = config.BASKET_PULSE_SEC

    # ------------------------------------------------------------------ 그리기
    def _draw(self, frame):
        self.renderer.draw_frame(frame)
        self.renderer.draw_contours(self.polys)
        self.renderer.draw_basket(self.pulse / config.BASKET_PULSE_SEC)

        if self.state != CALIBRATING:
            self.renderer.draw_ball(self.world.ball_pos)

        if self.state == CALIBRATING:
            self.renderer.draw_center_message(
                f"배경 인식 {max(0, int(self.timer) + 1)}",
                "화면에서 물건과 손을 치워주세요",
                progress=1.0 - max(0.0, self.timer) / config.CALIBRATION_COUNTDOWN,
            )
        elif self.state == RESULT and self.last_result:
            text, color = self.last_result
            self.renderer.draw_center_message(text, color=color)

        if self.debug:
            self.renderer.draw_mask(self.mask)

        self.renderer.draw_hud(
            self.success, self.fail, self.clock.get_fps(), self.debug, self.detector
        )

        if self.flash:
            color, remaining = self.flash
            self.renderer.draw_flash(color, remaining / config.FLASH_SEC)


def main():
    try:
        Game().run()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
