"""게임 전체의 튜닝 상수.

값 조정은 이 파일에서만 하면 되도록 모아두었다.
좌표는 전 구간 카메라 픽셀 기준(640x480, y축 아래 방향)으로 통일한다.
"""

import cv2

# --- 카메라 ---
CAM_INDEX = 0
CAM_BACKEND = cv2.CAP_DSHOW  # Windows에서 MSMF보다 초기화가 빠르다
CAMERA_WARMUP_FRAMES = 45    # 자동 노출이 수렴할 때까지 버릴 프레임 수
FRAME_W = 640
FRAME_H = 480

# --- 표시 ---
DISPLAY_SCALE = 1.5
WINDOW_W = int(FRAME_W * DISPLAY_SCALE)
WINDOW_H = int(FRAME_H * DISPLAY_SCALE)
TARGET_FPS = 60
WINDOW_TITLE = "Drop Ball - OpenCV Object Detective"

# --- 물리 ---
GRAVITY = (0.0, 900.0)  # y축이 아래 방향이므로 중력은 양수
BALL_RADIUS = 18.0  # 크게 할수록 쉬워지고 관통 가능성도 줄어든다
BALL_MASS = 1.0
BALL_ELASTICITY = 0.45
BALL_FRICTION = 0.5
MAX_BALL_SPEED = 1200.0  # 얇은 콜라이더 관통(터널링) 방지
PHYSICS_SUBSTEPS = 4     # dt를 나눠 여러 번 적분해 관통 확률을 더 낮춘다
WALL_RADIUS = 4.0
WALL_ELASTICITY = 0.4
WALL_FRICTION = 0.6

# --- 검출 ---
# 노이즈 관련 값은 조명과 배경에 크게 좌우된다. 게임 안에서 [ ] , . 키로
# 조절하면서 D 디버그 마스크로 확인하는 편이 빠르다.
DIFF_THRESHOLD = 38      # 배경과 이만큼 밝기가 다르면 물체로 인정
BLUR_KERNEL = (21, 21)
MORPH_KERNEL_SIZE = 7
# 팽창을 키우면 한 물체가 여러 조각으로 쪼개진 것이 하나로 합쳐져 윤곽이
# 훨씬 덜 떨린다(실측: 다각형 4.7->1.4개, 깜빡임 -66%). 대신 노이즈 덩어리도
# 같이 커지므로 MIN_CONTOUR_AREA를 함께 올려야 한다. 둘은 세트로 조절할 것.
DILATE_ITER = 4
MIN_CONTOUR_AREA = 6000.0  # 이보다 작은 덩어리는 노이즈로 간주 (DILATE_ITER와 세트)
MASK_SMOOTHING = 0.45      # 마스크 시간 평균 계수. 낮출수록 안정적이지만 반응이 느려진다
BRIGHTNESS_COMPENSATION = True  # 조명/자동노출로 화면 전체 밝기가 변해도 버티게 한다
BRIGHTNESS_SAMPLE_STEP = 6      # 밝기 보정량 추정 시 픽셀 건너뛰기 간격
MAX_CONTOURS = 6           # 큰 것부터 N개만 콜라이더로 (성능)
APPROX_EPS_RATIO = 0.008   # 윤곽선 단순화 강도
# 검출 파이프라인 전체가 프레임당 2ms 남짓(30fps 예산의 6%)이라 매 프레임 돌려도
# 여유롭다. 값을 올리면 부하는 줄지만 물체 움직임에 대한 반응이 그만큼 느려진다.
DETECT_EVERY_N_FRAMES = 1

# --- 장애물 콜라이더 ---
SEGMENT_RADIUS = 3.0
OBSTACLE_ELASTICITY = 0.55
OBSTACLE_FRICTION = 0.6

# --- 레이아웃 ---
SPAWN_X_RATIO = (0.72, 0.92)   # 공은 오른쪽 위에서 떨어진다
SPAWN_Y = -20.0
# 바구니는 왼쪽 아래. 왼쪽 벽에 딱 붙여 화면 끝과의 틈을 없앤다.
# 틈이 있으면 공이 그 사이로 빠져 사라져서 버그처럼 보인다.
BASKET_X_RATIO = (0.0, 0.26)
BASKET_TOP_RATIO = 0.78
BASKET_BOTTOM_RATIO = 0.92
BASKET_WALL_RADIUS = 5.0

BASKET_LEFT = FRAME_W * BASKET_X_RATIO[0]
BASKET_RIGHT = FRAME_W * BASKET_X_RATIO[1]
BASKET_TOP = FRAME_H * BASKET_TOP_RATIO
BASKET_BOTTOM = FRAME_H * BASKET_BOTTOM_RATIO


def basket_segments():
    """바구니를 이루는 선분 3개. 위가 열린 U자 형태."""
    left, right = BASKET_LEFT, BASKET_RIGHT
    top, bottom = BASKET_TOP, BASKET_BOTTOM
    return [
        ((left, top), (left, bottom)),
        ((left, bottom), (right, bottom)),
        ((right, bottom), (right, top)),
    ]


# --- 게임 진행 ---
CALIBRATION_COUNTDOWN = 3.0
RESULT_DISPLAY_SEC = 0.8
BALL_TIMEOUT_SEC = 12.0   # 최후의 안전장치
STILL_DISTANCE = 12.0     # 이 거리 안에 머물면 멈춘 것으로 본다
STILL_TIMEOUT_SEC = 1.5   # 물체 위에 얹혀 멈춘 공을 빠르게 회수
LOST_MARGIN = 60.0       # 화면 아래로 이만큼 더 내려가면 실패

# --- UI ---
BG_DIM = 0.62              # 배경 영상 밝기 배율. 낮출수록 게임 요소가 또렷해진다
VIGNETTE_INNER = 0.45      # 이 반경까지는 디밍만, 바깥부터 비네트 (정규화 거리)
VIGNETTE_STRENGTH = 0.55   # 화면 모서리에서 추가로 어두워지는 정도
BALL_TRAIL_LENGTH = 14     # 공 잔상 개수. 궤적이 보여야 튕긴 방향을 읽을 수 있다
FLASH_SEC = 0.45           # 성공/실패 시 화면 테두리 플래시 지속
BASKET_PULSE_SEC = 0.6     # 성공 직후 바구니가 밝게 빛나는 시간
PANEL_RADIUS = 12

# --- 색 (pygame RGB / 알파가 필요한 것은 RGBA) ---
# 배경을 BG_DIM만큼 눌러두었으므로 전경은 채도 높은 네온 계열로 띄운다.
COLOR_BALL = (255, 122, 66)
COLOR_BALL_CORE = (255, 214, 164)      # 좌상단 하이라이트
COLOR_BALL_EDGE = (255, 246, 238)
COLOR_BALL_GLOW = (255, 110, 40)       # 가산 합성용이라 알파 없음
COLOR_BALL_TRAIL = (255, 146, 86)

COLOR_BASKET = (86, 204, 242)
COLOR_BASKET_GLOW = (32, 120, 168)
COLOR_BASKET_FILL = (86, 204, 242, 46)

COLOR_CONTOUR = (0, 255, 200)
COLOR_CONTOUR_SOFT = (0, 150, 124)     # 외곽선 글로우
COLOR_CONTOUR_FILL = (0, 255, 200, 40)

COLOR_PANEL = (13, 17, 26, 176)
COLOR_PANEL_EDGE = (255, 255, 255, 34)
COLOR_KEYCAP = (52, 60, 78)            # 화면에 직접 그리므로 불투명 RGB
COLOR_KEYCAP_EDGE = (112, 124, 148)
COLOR_TRACK = (58, 66, 84)             # 진행 바 바닥

COLOR_TEXT = (245, 247, 250)
COLOR_TEXT_DIM = (170, 180, 196)
COLOR_SUCCESS = (116, 231, 150)
COLOR_FAIL = (240, 110, 110)
COLOR_SHADOW = (0, 0, 0)
