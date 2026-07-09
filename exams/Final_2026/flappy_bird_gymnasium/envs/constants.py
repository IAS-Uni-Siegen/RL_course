############################ Speed and Acceleration ############################
PIPE_VEL_X = -4

PLAYER_MAX_VEL_Y = 10  # max vel along Y, max descend speed
PLAYER_MIN_VEL_Y = -8  # min vel along Y, max ascend speed

PLAYER_ACC_Y = 1  # players downward acceleration
PLAYER_VEL_ROT = 3  # angular speed

PLAYER_FLAP_ACC = -13  # players speed on flapping
################################################################################

################################## Dimensions ##################################
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24
PLAYER_PRIVATE_ZONE = (max(PLAYER_WIDTH, PLAYER_HEIGHT) + 30) / 2

PIPE_WIDTH = 52
PIPE_HEIGHT = 320

BASE_WIDTH = 336
BASE_HEIGHT = 112

BACKGROUND_WIDTH = 288
BACKGROUND_HEIGHT = 512
################################################################################

#: Player's rotation threshold.
PLAYER_ROT_THR = 20

#: Color to fill the surface's background when no background image was loaded.
FILL_BACKGROUND_COLOR = (200, 200, 200)
