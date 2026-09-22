"""pymunk 물리 공간.

좌표는 카메라 픽셀 그대로(y축 아래 방향) 사용한다. pymunk는 축 방향에
무관하게 동작하므로 중력을 (0, +900)으로 주면 y-down이 그대로 성립하고,
OpenCV/pygame과의 좌표 변환 코드가 전혀 필요 없다.
"""

import random

import cv2
import pymunk

import config


class World:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = config.GRAVITY
        self.obstacle_shapes = []
        self.ball_body = None
        self.ball_shape = None
        self.ball_age = 0.0
        self.still_time = 0.0
        self._still_origin = (0.0, 0.0)
        self._build_static()
        self.spawn_ball()

    # ------------------------------------------------------------------ 정적 지형
    def _build_static(self):
        static = self.space.static_body
        w, h = config.FRAME_W, config.FRAME_H

        # 좌우 벽. 위로 연장해 스폰 직후 화면 밖으로 새지 않게 한다.
        for a, b in (((0, -h), (0, h)), ((w, -h), (w, h))):
            seg = pymunk.Segment(static, a, b, config.WALL_RADIUS)
            seg.elasticity = config.WALL_ELASTICITY
            seg.friction = config.WALL_FRICTION
            self.space.add(seg)

        # 바구니. 들어간 공이 튀어나가지 않도록 탄성을 낮추고 마찰을 높인다.
        for a, b in config.basket_segments():
            seg = pymunk.Segment(static, a, b, config.BASKET_WALL_RADIUS)
            seg.elasticity = 0.15
            seg.friction = 0.9
            self.space.add(seg)

    # ------------------------------------------------------------------ 공
    def spawn_ball(self):
        if self.ball_body is not None:
            self.space.remove(self.ball_body, self.ball_shape)

        moment = pymunk.moment_for_circle(config.BALL_MASS, 0, config.BALL_RADIUS)
        body = pymunk.Body(config.BALL_MASS, moment)
        low, high = config.SPAWN_X_RATIO
        body.position = (random.uniform(low, high) * config.FRAME_W, config.SPAWN_Y)

        shape = pymunk.Circle(body, config.BALL_RADIUS)
        shape.elasticity = config.BALL_ELASTICITY
        shape.friction = config.BALL_FRICTION

        self.space.add(body, shape)
        self.ball_body, self.ball_shape = body, shape
        self.ball_age = 0.0
        self.still_time = 0.0
        self._still_origin = body.position

    @property
    def ball_pos(self):
        return self.ball_body.position

    # ------------------------------------------------------------------ 장애물
    def sync_obstacles(self, polys):
        """검출된 다각형으로 정적 콜라이더를 통째로 갈아끼운다."""
        if self.obstacle_shapes:
            self.space.remove(*self.obstacle_shapes)
            self.obstacle_shapes.clear()

        static = self.space.static_body
        ball_x, ball_y = self.ball_body.position

        for poly in polys:
            # 공이 이미 내부에 들어와 있는 도형은 이번 프레임만 건너뛴다.
            # 그대로 생성하면 pymunk가 겹친 공을 강하게 밀어내 순간이동이 생긴다.
            if cv2.pointPolygonTest(poly, (float(ball_x), float(ball_y)), False) >= 0:
                continue

            pts = [(float(p[0]), float(p[1])) for p in poly]
            for i in range(len(pts)):
                seg = pymunk.Segment(
                    static, pts[i], pts[(i + 1) % len(pts)], config.SEGMENT_RADIUS
                )
                seg.elasticity = config.OBSTACLE_ELASTICITY
                seg.friction = config.OBSTACLE_FRICTION
                self.obstacle_shapes.append(seg)

        if self.obstacle_shapes:
            self.space.add(*self.obstacle_shapes)

    # ------------------------------------------------------------------ 진행
    def step(self, dt):
        sub = dt / config.PHYSICS_SUBSTEPS
        for _ in range(config.PHYSICS_SUBSTEPS):
            self.space.step(sub)
            velocity = self.ball_body.velocity
            if velocity.length > config.MAX_BALL_SPEED:
                self.ball_body.velocity = velocity.normalized() * config.MAX_BALL_SPEED
        self.ball_age += dt

        # 물체 위에 얹혀 멈춘 공은 빠르게 회수한다. 없으면 12초를 기다려야 한다.
        # 속도가 아니라 실제 이동 거리로 재는 이유: 콜라이더를 매 프레임 새로
        # 만들 때 멈춰 있는 공도 미세하게 튕겨 속도가 순간적으로 치솟는다.
        # 속도로 재면 그때마다 타이머가 초기화돼 회수가 4초까지 늘어졌다.
        if (self.ball_body.position - self._still_origin).length > config.STILL_DISTANCE:
            self._still_origin = self.ball_body.position
            self.still_time = 0.0
        else:
            self.still_time += dt

    def ball_in_basket(self):
        x, y = self.ball_body.position
        inset = config.BALL_RADIUS * 0.5
        return (
            config.BASKET_LEFT + inset < x < config.BASKET_RIGHT - inset
            and config.BASKET_TOP < y < config.BASKET_BOTTOM
        )

    def ball_lost(self):
        """화면 아래로 빠졌거나, 멈춰버렸거나, 제한 시간이 다 된 경우."""
        if self.ball_body.position.y > config.FRAME_H + config.LOST_MARGIN:
            return True
        if self.still_time > config.STILL_TIMEOUT_SEC:
            return True
        return self.ball_age > config.BALL_TIMEOUT_SEC
