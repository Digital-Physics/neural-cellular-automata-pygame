import pygame
import yaml
import os

SCREEN_WIDTH = 750
SCREEN_HEIGHT = SCREEN_WIDTH
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
FPS = 30
last_time_check = pygame.time.get_ticks()  # for animation update interval measurement
last_time_check_dialogue = pygame.time.get_ticks()
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
dialogue_counter = 0
dialogue_counter_flag = False
ANIMATION_INTERVAL = 75
DIALOGUE_INTERVAL = 4000

def resource_path(relative_path):
    base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# load dialogue
with open(resource_path("dialogue/dialogue.yml"), "r") as temp_file:
    dialogue = yaml.load(temp_file, Loader=yaml.FullLoader)

pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.mixer.init()
pygame.init()
pygame.mixer.music.load(resource_path("audio/level0_music.mp3"))
pygame.mixer.music.set_volume(0.08)
pygame.mixer.music.play(-1, 0.0, 1000)

speech_sfx = pygame.mixer.Sound(resource_path("audio/speech.wav"))
speech_sfx.set_volume(0.05)
door_sfx = pygame.mixer.Sound(resource_path("audio/door.wav"))
door_sfx.set_volume(0.2)
success_sfx = pygame.mixer.Sound(resource_path("audio/success.wav"))
success_sfx.set_volume(0.2)

escher3_img = pygame.image.load(resource_path("img/Background/escher_wallpaper_computer3.png")).convert_alpha()
escher3_img = pygame.transform.scale(escher3_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
escher4_img = pygame.image.load(resource_path("img/Background/escher_wallpaper_computer4.png")).convert_alpha()
escher4_img = pygame.transform.scale(escher4_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

# load custom font
font = pygame.font.Font(resource_path("fonts/kongtext.ttf"), 20)
