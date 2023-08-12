import pygame
import pygame.locals
import global_vars as gv
import torch
import nca_model as nca
from PIL import Image
import numpy as np
import various_functions as f
# probably shouldn't use torch tensor if we don't need to backprop anymore?


class NCA_room():
    def __init__(self):
        self.control = pygame.Rect(3, 3, 10, 10)
        self.loaded_model_state = torch.load("nca_model/checkpoint.pth")
        self.model = nca.CAModel(n_channels=16, device="cpu")
        self.model.load_state_dict(self.loaded_model_state)
        self.seed = nca.make_seed(32, 16).to("cpu")
        self.x_eval = self.seed.clone()
        self.x_eval_out = nca.to_rgb(self.x_eval[:, :4].detach().cpu())*255
        self.x_eval_out = self.x_eval_out.permute(0,3,2,1).numpy()[0]
        self.cleared = False
        self.intro_complete = False
        self.target = None
        gv.last_time_check_dialogue = pygame.time.get_ticks()
        self.dialogue_change_state = 0
        self.last_time_check = pygame.time.get_ticks()
        self.key_to_function = {
            pygame.K_LEFT: (lambda x: x.translate('left')),
            pygame.K_RIGHT: (lambda x: x.translate('right')),
            pygame.K_UP: (lambda x: x.translate('up')),
            pygame.K_DOWN: (lambda x: x.translate('down')),
            pygame.K_SLASH: (lambda x: x.translate('clear')),
            pygame.K_PERIOD: (lambda x: x.translate("add"))}
        self.load_image()
        self.last_time = 0

    def translate(self, direction):
        if direction == "right":
            self.control.x = (self.control.x+1)%29  # 59
        elif direction == "left":
            self.control.x = (self.control.x-1)%29  # 59
        elif direction == "up":
            self.control.y = (self.control.y-1)%29  # 59
        elif direction == "down":
            self.control.y = (self.control.y+1)%29  # 59
        elif direction == "clear":
            self.cleared = True
        elif direction == "add":
            nca.make_seed(32, 16, pos=(self.control.x, self.control.y), tensor=self.x_eval)

    def load_image(self):
        img = Image.open(gv.resource_path('img/93.png'))
        img = np.float32(img) / 255.0
        img[..., :3] *= img[..., 3:]
        self.target = torch.from_numpy(img).permute(2, 0, 1)[None, ...]

    def corruptible_dialogue(self):
        # use this loss distance between the target image and the current image to distort the dialogue
        # in theory, we would like a loss that is invariant to translation (and rotation, but rotation isn't possible here)
        if self.dialogue_change_state == 0:
            loss = ((self.target - self.x_eval[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
            loss = loss.mean()

            # stable error rate on fully formed NCA ~ 0.0003
            flip_n = []
            for line in gv.dialogue["notkhatchig"][gv.dialogue_counter % len(gv.dialogue["notkhatchig"])]:
                length_to_flip = min(int(200*(loss.item()-0.0003)), len(line))
                flip_n.append(length_to_flip)

            temp_dialogue = gv.dialogue["notkhatchig"][gv.dialogue_counter % len(gv.dialogue["notkhatchig"])][:]

            for i in range(len(temp_dialogue)):
                for char_idx in range(0, flip_n[i], 2):
                    temp_dialogue[i] = temp_dialogue[i][:char_idx] + "?" + temp_dialogue[i][char_idx + 1:]
        elif self.dialogue_change_state == 1:
            temp_dialogue = gv.dialogue["erased_notkhatchig"][gv.dialogue_counter % len(gv.dialogue["erased_notkhatchig"])][:]
        elif self.dialogue_change_state == 2:
            temp_dialogue = gv.dialogue["success_notkhatchig"][gv.dialogue_counter % len(gv.dialogue["success_notkhatchig"])][:]

        f.draw_dialogue(temp_dialogue, 180, 250)
        if pygame.time.get_ticks() - gv.last_time_check_dialogue > gv.DIALOGUE_INTERVAL:
            gv.dialogue_counter_flag = True
            gv.dialogue_counter += 1  
            gv.speech_sfx.play()
            gv.last_time_check_dialogue = pygame.time.get_ticks()

    def run(self):
        """ self-contained loop in main game loop """
        img_flip = True

        frame = 0
        # while gv.world_level == "neural_cellular_automata_room":
        while True:
            gv.clock.tick(gv.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.locals.K_SPACE:
                        gv.door_sfx.play()
                        self.__init__()
                        self.run()
                    if event.key in self.key_to_function:
                        # dictionary returns a method (run on itself)
                        self.key_to_function[event.key](self)

            self.x_eval = self.model(self.x_eval)
            self.x_eval_out = nca.to_rgb(self.x_eval[:, :4].detach().cpu())*255
            self.x_eval_out = self.x_eval_out.permute(0, 3, 2, 1).numpy()[0]
            self.x_eval = nca.update_rgb_tensor(self.x_eval, self.cleared, self.control.x, self.control.y)
            erased_notkhatchig = nca.check_erased(self.x_eval)

            # erased and regrown
            if not erased_notkhatchig and self.dialogue_change_state == 1:
                gv.success_sfx.play()
                # gv.to_do["erase notkhatchig"] = True
                self.dialogue_change_state = 2
            elif erased_notkhatchig:
                if self.dialogue_change_state != 1:  # just erased
                    gv.success_sfx.play()
                    self.dialogue_change_state = 1

            self.cleared = False

            if pygame.time.get_ticks() - self.last_time_check > gv.ANIMATION_INTERVAL:
                self.last_time_check = pygame.time.get_ticks()
                img_flip = not img_flip

            if img_flip:
                gv.screen.blit(gv.escher3_img, (0, 0))
            else:
                gv.screen.blit(gv.escher4_img, (0, 0))

            surf = pygame.surfarray.make_surface(self.x_eval_out)
            surf = pygame.transform.scale(surf, (163, 163))
            gv.screen.blit(surf, (116, 310))

            pygame.draw.rect(surface=gv.screen, color=(125,55,200),
                             rect=pygame.Rect(self.control.x*5+116, self.control.y*5+310, 25, 25), width=1)

            self.corruptible_dialogue()

            pygame.display.update()

            # pygame.image.save(gv.screen, f"frames/frame{frame}.jpeg")
            # frame += 1


def create():
    room = NCA_room()
    room.run()

if __name__ == "__main__":
    create()
