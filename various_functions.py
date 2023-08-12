import pygame
import global_vars as gv


def draw_dialogue(text, x, y):
    # set dimensions of dialogue box
    dialogue_surface = pygame.Surface((max([len(line) for line in text])*10, len(text)*13))

    dialogue_font = pygame.font.Font(gv.resource_path("fonts/kongtext.ttf"), 10)

    for i, line in enumerate(text):
        text_surface = dialogue_font.render(line, True, gv.WHITE)
        dialogue_surface.blit(text_surface, (0,i*15))
    dialogue_rect = dialogue_surface.get_rect(midbottom=(x,y))

    bg_rect = dialogue_rect.copy()
    bg_rect.inflate_ip(10, 10)  # enlarge the copy in position

    frame_rect = bg_rect.copy()
    frame_rect.inflate_ip(4, 4)

    pygame.draw.rect(gv.screen, gv.WHITE, frame_rect)
    pygame.draw.rect(gv.screen, gv.BLACK, bg_rect)
    gv.screen.blit(dialogue_surface, dialogue_rect)


