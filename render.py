"""pygame 렌더링. 웹캠 영상을 배경으로 깔고 그 위에 게임 요소를 그린다.

영상 위에 그리는 UI라 배경이 밝으면 전경이 통째로 묻힌다. 그래서 배경은
한 번 눌러두고(BG_DIM + 비네트), 전경은 글로우와 반투명 패널로 대비를 만든다.
"""

import math
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
import pygame

import config

# 한글 표시용 폰트 후보. 앞에서부터 설치된 것을 고른다.
_FONT_CANDIDATES = ("malgungothic", "nanumgothic", "gulim", "dotum", "batang")

_KEY_HINTS = (
    ("B", "배경 재보정"), ("D", "디버그"), ("[ ]", "민감도"),
    (", .", "최소 크기"), ("R", "점수 초기화"), ("ESC", "종료"),
)


def _pick_font():
    installed = set(pygame.font.get_fonts())
    for name in _FONT_CANDIDATES:
        if name in installed:
            return name
    return None  # pygame 기본 폰트로 대체 (한글은 깨질 수 있음)


def _scaled(point):
    """카메라 좌표 -> 화면 좌표."""
    return (int(point[0] * config.DISPLAY_SCALE), int(point[1] * config.DISPLAY_SCALE))


def _px(length):
    """카메라 기준 길이 -> 화면 기준 길이."""
    return max(1, int(length * config.DISPLAY_SCALE))


@lru_cache(maxsize=64)
def _panel(size, fill, border, radius):
    """반투명 둥근 패널. 인자가 같으면 캐시된 서피스를 재사용한다."""
    surf = pygame.Surface(size, pygame.SRCALPHA)
    rect = surf.get_rect()
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, width=1, border_radius=radius)
    return surf


@lru_cache(maxsize=16)
def _glow(radius, color, strength=0.55, layers=9):
    """가산 합성(BLEND_RGB_ADD)용 원형 글로우.

    가산 합성은 알파를 무시하므로 감쇠를 알파가 아니라 색의 밝기로 만든다.
    바깥일수록 어두운 동심원을 큰 것부터 덮어 그려 그라데이션을 얻는다.
    """
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    center = (radius, radius)
    for i in range(layers, 0, -1):
        t = i / layers
        gain = (1.0 - t) ** 2 * strength
        rgb = tuple(min(255, int(channel * gain)) for channel in color)
        pygame.draw.circle(surf, (*rgb, 255), center, max(1, int(radius * t)))
    return surf


@lru_cache(maxsize=4)
def _edge_flash(size, color):
    """화면 테두리에서 안쪽으로 사그라드는 색 띠."""
    width, height = size
    surf = pygame.Surface(size, pygame.SRCALPHA)
    band = 54
    for i in range(band, 0, -1):
        alpha = int(150 * (1.0 - i / band) ** 2)
        pygame.draw.rect(
            surf, (*color, alpha),
            (i, i, width - 2 * i, height - 2 * i), width=2, border_radius=10,
        )
    return surf


class Renderer:
    def __init__(self):
        self.screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
        pygame.display.set_caption(config.WINDOW_TITLE)
        # 매 프레임 서피스를 새로 만들면 할당 비용이 크다. 두 장을 미리 잡아두고
        # 픽셀만 덮어쓰면 draw_frame이 3.4ms에서 1.5ms로 줄어든다.
        self._src = pygame.Surface((config.FRAME_W, config.FRAME_H))
        self._dst = pygame.Surface((config.WINDOW_W, config.WINDOW_H))
        self._mask_surface = pygame.Surface((config.FRAME_W // 4, config.FRAME_H // 4))

        name = _pick_font()
        self.font_big = pygame.font.SysFont(name, 44, bold=True)
        self.font = pygame.font.SysFont(name, 22, bold=True)
        self.font_num = pygame.font.SysFont(name, 30, bold=True)
        self.font_label = pygame.font.SysFont(name, 14, bold=True)
        self.font_small = pygame.font.SysFont(name, 13)

        # 카드 높이는 실제 폰트 높이에서 뽑는다. 상수로 박으면 대체 폰트가
        # 잡혔을 때 숫자 아래가 잘린다.
        self._caption_h = self.font_label.get_height()
        self._card_h = 10 + self._caption_h + 2 + self.font_num.get_height() + 10

        self._vignette = self._build_vignette()
        # 반투명 채움용 스크래치. 매 프레임 새로 할당하지 않고 지우고 돌려쓴다.
        self._overlay = pygame.Surface((config.WINDOW_W, config.WINDOW_H), pygame.SRCALPHA)
        self._trail = deque(maxlen=config.BALL_TRAIL_LENGTH)

    @staticmethod
    def _build_vignette():
        """카메라 해상도 크기의 밝기 마스크(uint8 BGR). 시작할 때 한 번만 만든다."""
        h, w = config.FRAME_H, config.FRAME_W
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - (w - 1) / 2.0) / ((w - 1) / 2.0)
        ny = (yy - (h - 1) / 2.0) / ((h - 1) / 2.0)
        radius = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
        t = np.clip((radius - config.VIGNETTE_INNER) / (1.0 - config.VIGNETTE_INNER), 0.0, 1.0)
        gain = 1.0 - config.VIGNETTE_STRENGTH * t ** 1.6
        mask = np.clip(gain * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ 배경
    def draw_frame(self, frame):
        # 표시용 복사본만 어둡게 한다. 검출기는 game.py에서 원본을 그대로 받는다.
        dimmed = cv2.convertScaleAbs(frame, alpha=config.BG_DIM)
        dimmed = cv2.multiply(dimmed, self._vignette, scale=1.0 / 255.0)
        rgb = cv2.cvtColor(dimmed, cv2.COLOR_BGR2RGB)
        # OpenCV 프레임은 (h, w, 3)인데 pygame은 (w, h, 3)을 기대한다.
        # swapaxes는 뷰만 만들어서 cv2.transpose로 복사하는 것보다 빠르다.
        pygame.surfarray.blit_array(self._src, rgb.swapaxes(0, 1))
        if config.DISPLAY_SCALE == 1.0:
            self.screen.blit(self._src, (0, 0))
        else:
            pygame.transform.scale(self._src, (config.WINDOW_W, config.WINDOW_H), self._dst)
            self.screen.blit(self._dst, (0, 0))

    # ------------------------------------------------------------------ 게임 요소
    def draw_contours(self, polys):
        """검출된 윤곽을 그린다.

        이 피드백이 없으면 플레이어는 자기 물건이 콜라이더로 등록됐는지
        알 수 없어서 게임이 성립하지 않는다. 안쪽을 옅게 채워서, 선만 그릴 때보다
        "이 면이 벽"이라는 게 분명히 보이게 한다.
        """
        shapes = []
        self._overlay.fill((0, 0, 0, 0))
        for poly in polys:
            points = [_scaled(p) for p in poly]
            if len(points) >= 3:
                shapes.append(points)
                pygame.draw.polygon(self._overlay, config.COLOR_CONTOUR_FILL, points)
        if not shapes:
            return

        self.screen.blit(self._overlay, (0, 0))
        for points in shapes:
            pygame.draw.polygon(self.screen, config.COLOR_CONTOUR_SOFT, points, 7)
            pygame.draw.polygon(self.screen, config.COLOR_CONTOUR, points, 2)

    def draw_basket(self, pulse=0.0):
        """pulse: 성공 직후 1 -> 0으로 줄어드는 값. 잠깐 더 밝게 빛낸다."""
        center_x = (config.BASKET_LEFT + config.BASKET_RIGHT) / 2

        self._overlay.fill((0, 0, 0, 0))
        top_left = _scaled((config.BASKET_LEFT, config.BASKET_TOP))
        bottom_right = _scaled((config.BASKET_RIGHT, config.BASKET_BOTTOM))
        rect = pygame.Rect(
            top_left,
            (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]),
        )
        r, g, b, a = config.COLOR_BASKET_FILL
        pygame.draw.rect(
            self._overlay, (r, g, b, min(255, int(a * (1.0 + 2.5 * pulse)))),
            rect, border_radius=6,
        )
        self._draw_chevrons(center_x)
        self.screen.blit(self._overlay, (0, 0))

        glow_width = _px(8) + int(_px(6) * pulse)
        for start, end in config.basket_segments():
            pygame.draw.line(
                self.screen, config.COLOR_BASKET_GLOW,
                _scaled(start), _scaled(end), glow_width + _px(5),
            )
        for start, end in config.basket_segments():
            pygame.draw.line(self.screen, config.COLOR_BASKET, _scaled(start), _scaled(end), _px(4))

        label = self.font_label.render("GOAL", True, config.COLOR_BASKET)
        pill = _panel(
            (label.get_width() + 22, label.get_height() + 10),
            config.COLOR_PANEL, config.COLOR_PANEL_EDGE, 11,
        )
        # 입구 위, 화살표 바로 위에 둔다. 바구니 안에 두면 안착한 공과 겹치고,
        # 아래에 두면 화면 맨 끝에 붙어 잘릴 위험이 있다.
        pill_rect = pill.get_rect(center=_scaled((center_x, config.BASKET_TOP - 52)))
        self.screen.blit(pill, pill_rect)
        self.screen.blit(label, label.get_rect(center=pill_rect.center))

    def _draw_chevrons(self, center_x):
        """바구니 입구 위로 흘러내리는 화살표. 어디로 넣어야 하는지 가리킨다."""
        phase = (pygame.time.get_ticks() % 1400) / 1400.0
        for i in range(3):
            t = (phase + i / 3.0) % 1.0
            y = config.BASKET_TOP - 34 + t * 24
            alpha = int(190 * math.sin(math.pi * t))
            if alpha <= 4:
                continue
            points = [_scaled(p) for p in
                      ((center_x - 13, y), (center_x, y + 8), (center_x + 13, y))]
            pygame.draw.lines(self._overlay, (*config.COLOR_BASKET, alpha), False, points, 4)

    def draw_ball(self, position):
        radius = _px(config.BALL_RADIUS)
        self._trail.append((float(position[0]), float(position[1])))

        # 잔상: 오래된 것일수록 작고 흐리게. 어디서 어떻게 튕겼는지 눈으로 따라갈 수 있다.
        count = len(self._trail)
        if count > 1:
            self._overlay.fill((0, 0, 0, 0))
            for i, point in enumerate(self._trail):
                t = (i + 1) / count
                alpha = int(110 * t ** 2)
                if alpha <= 3:
                    continue
                pygame.draw.circle(
                    self._overlay, (*config.COLOR_BALL_TRAIL, alpha),
                    _scaled(point), max(1, int(radius * (0.25 + 0.6 * t))),
                )
            self.screen.blit(self._overlay, (0, 0))

        center = _scaled(position)
        glow = _glow(radius * 4, config.COLOR_BALL_GLOW, 0.8)
        self.screen.blit(glow, glow.get_rect(center=center), special_flags=pygame.BLEND_RGB_ADD)
        pygame.draw.circle(self.screen, config.COLOR_BALL, center, radius)
        pygame.draw.circle(
            self.screen, config.COLOR_BALL_CORE,
            (center[0] - radius // 3, center[1] - radius // 3), max(2, radius // 3),
        )
        pygame.draw.circle(self.screen, config.COLOR_BALL_EDGE, center, radius, 2)

    def reset_ball_trail(self):
        """리스폰 시 호출. 없으면 옛 궤적과 새 공이 선으로 이어져 보인다."""
        self._trail.clear()

    # ------------------------------------------------------------------ HUD
    def _text(self, surface_font, text, pos, color, center=False):
        shadow = surface_font.render(text, True, config.COLOR_SHADOW)
        label = surface_font.render(text, True, color)
        rect = label.get_rect(center=pos) if center else label.get_rect(topleft=pos)
        # 패널 밖에서도 읽히도록 그림자를 한 겹 깐다.
        self.screen.blit(shadow, rect.move(2, 2))
        self.screen.blit(label, rect)

    def _stat_card(self, topleft, caption, value, accent):
        """작은 라벨 + 큰 숫자 카드. 오른쪽 끝 x를 돌려준다."""
        width, height = 118, self._card_h
        self.screen.blit(
            _panel((width, height), config.COLOR_PANEL, config.COLOR_PANEL_EDGE, config.PANEL_RADIUS),
            topleft,
        )
        x, y = topleft
        pygame.draw.rect(self.screen, accent, (x + 10, y + 12, 4, height - 24), border_radius=2)
        self.screen.blit(self.font_label.render(caption, True, config.COLOR_TEXT_DIM), (x + 24, y + 10))
        self.screen.blit(
            self.font_num.render(value, True, accent), (x + 23, y + 12 + self._caption_h)
        )
        return x + width

    def _chip(self, text, topright, color):
        label = self.font_label.render(text, True, color)
        size = (label.get_width() + 22, label.get_height() + 12)
        chip = _panel(size, config.COLOR_PANEL, config.COLOR_PANEL_EDGE, size[1] // 2)
        rect = chip.get_rect(topright=topright)
        self.screen.blit(chip, rect)
        self.screen.blit(label, label.get_rect(center=rect.center))
        return rect

    def draw_hud(self, success, fail, fps, debug, detector=None):
        x = self._stat_card((18, 14), "성공", str(success), config.COLOR_SUCCESS) + 10
        self._stat_card((x, 14), "실패", str(fail), config.COLOR_FAIL)

        fps_rect = self._chip(f"{fps:.0f} FPS", (config.WINDOW_W - 18, 16), config.COLOR_TEXT_DIM)
        if debug:
            self._chip("DEBUG", (fps_rect.left - 8, 16), config.COLOR_CONTOUR)
            if detector is not None:
                self._chip(
                    f"민감도 {detector.diff_threshold}   최소크기 {int(detector.min_area)}",
                    (config.WINDOW_W - 18, 232), config.COLOR_CONTOUR,
                )

        self._draw_key_hints()

    def _draw_key_hints(self):
        gap, pad = 18, 14
        items, total = [], 0
        for key, desc in _KEY_HINTS:
            key_surf = self.font_label.render(key, True, config.COLOR_TEXT)
            desc_surf = self.font_small.render(desc, True, config.COLOR_TEXT_DIM)
            cap_width = key_surf.get_width() + 16
            items.append((key_surf, desc_surf, cap_width))
            total += cap_width + 6 + desc_surf.get_width()
        total += gap * (len(items) - 1)

        height = 40
        bar = _panel((total + pad * 2, height), config.COLOR_PANEL, config.COLOR_PANEL_EDGE, height // 2)
        bar_rect = bar.get_rect(midbottom=(config.WINDOW_W // 2, config.WINDOW_H - 14))
        self.screen.blit(bar, bar_rect)

        x, center_y = bar_rect.x + pad, bar_rect.centery
        for key_surf, desc_surf, cap_width in items:
            cap = pygame.Rect(x, center_y - 11, cap_width, 22)
            pygame.draw.rect(self.screen, config.COLOR_KEYCAP, cap, border_radius=6)
            pygame.draw.rect(self.screen, config.COLOR_KEYCAP_EDGE, cap, width=1, border_radius=6)
            self.screen.blit(key_surf, key_surf.get_rect(center=cap.center))
            x += cap_width + 6
            self.screen.blit(desc_surf, desc_surf.get_rect(midleft=(x, center_y)))
            x += desc_surf.get_width() + gap

    def draw_center_message(self, main, sub=None, color=None, progress=None):
        """progress(0~1)를 주면 패널 아래쪽에 카운트다운 진행 바를 함께 그린다."""
        color = color or config.COLOR_TEXT
        center_x, center_y = config.WINDOW_W // 2, config.WINDOW_H // 2
        bar_h, bar_gap = 8, 20

        main_surf = self.font_big.render(main, True, color)
        sub_surf = self.font.render(sub, True, config.COLOR_TEXT_DIM) if sub else None

        content_w = main_surf.get_width()
        content_h = main_surf.get_height()
        if sub_surf:
            content_w = max(content_w, sub_surf.get_width())
            content_h += sub_surf.get_height() + 12
        if progress is not None:
            content_w = max(content_w, 240)
            content_h += bar_gap + bar_h

        panel = _panel(
            (content_w + 72, content_h + 52), config.COLOR_PANEL, config.COLOR_PANEL_EDGE, 20
        )
        panel_rect = panel.get_rect(center=(center_x, center_y))
        self.screen.blit(panel, panel_rect)

        y = panel_rect.top + 26
        self.screen.blit(main_surf, main_surf.get_rect(midtop=(center_x, y)))
        y += main_surf.get_height()
        if sub_surf:
            y += 12
            self.screen.blit(sub_surf, sub_surf.get_rect(midtop=(center_x, y)))
            y += sub_surf.get_height()

        if progress is not None:
            y += bar_gap
            track = pygame.Rect(center_x - content_w // 2, y, content_w, bar_h)
            pygame.draw.rect(self.screen, config.COLOR_TRACK, track, border_radius=bar_h // 2)
            filled = int(content_w * max(0.0, min(1.0, progress)))
            if filled > 0:
                pygame.draw.rect(
                    self.screen, color,
                    (track.x, track.y, max(filled, bar_h), bar_h), border_radius=bar_h // 2,
                )

    def draw_flash(self, color, amount):
        """결과 직후 화면 테두리 플래시. amount는 1 -> 0으로 줄어드는 남은 비율."""
        if amount <= 0:
            return
        surf = _edge_flash((config.WINDOW_W, config.WINDOW_H), color)
        surf.set_alpha(int(255 * min(1.0, amount)))
        self.screen.blit(surf, (0, 0))

    def draw_mask(self, mask):
        """디버그용 이진 마스크 축소 표시. 임계값 튜닝에 쓴다."""
        if mask is None:
            return
        small = cv2.resize(mask, (config.FRAME_W // 4, config.FRAME_H // 4))
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        pygame.surfarray.blit_array(self._mask_surface, rgb.swapaxes(0, 1))
        surface = self._mask_surface
        width, height = surface.get_size()

        pad, caption_h = 8, 20
        panel = _panel(
            (width + pad * 2, height + pad * 2 + caption_h),
            config.COLOR_PANEL, config.COLOR_PANEL_EDGE, 10,
        )
        rect = panel.get_rect(topright=(config.WINDOW_W - 18, 60))
        self.screen.blit(panel, rect)
        self.screen.blit(
            self.font_label.render("MASK", True, config.COLOR_TEXT_DIM), (rect.x + pad + 2, rect.y + 6)
        )
        origin = (rect.x + pad, rect.y + pad + caption_h)
        self.screen.blit(surface, origin)
        pygame.draw.rect(self.screen, config.COLOR_CONTOUR_SOFT, (*origin, width, height), 1)
