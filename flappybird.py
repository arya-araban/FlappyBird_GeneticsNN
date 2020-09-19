#! /usr/bin/env python3

"""Flappy Bird, implemented using Pygame."""
import copy
import sys

from nn import NeuralNetwork
import math
import os
import random
import time
import keyboard
from random import randint
from collections import deque

import pygame
from pygame.locals import *
import tensorflow as tf

FPS = 60
TOTAL = 50  # total number of birds
FRAME_SKIPS = 15  # all birds think once every x frames (keep this between 1 and 20)
nm = 0
ANIMATION_SPEED = 0.18  # pixels per millisecond
WIN_WIDTH = 284 * 2  # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512


class Bird(pygame.sprite.Sprite):
    """Represents the bird controlled by the player.

    The bird is the 'hero' of this game.  The player can make it climb
    (ascend quickly), otherwise it sinks (descends more slowly).  It must
    pass through the space in between pipes (for every pipe passed, one
    point is scored); if it crashes into a pipe, the game ends.

    Attributes:
    x: The bird's X coordinate.
    y: The bird's Y coordinate
    msec_to_climb: The number of milliseconds left to climb, where a
        complete climb lasts Bird.CLIMB_DURATION milliseconds.

    Constants:
    WIDTH: The width, in pixels, of the bird's image.
    HEIGHT: The height, in pixels, of the bird's image.
    SINK_SPEED: With which speed, in pixels per millisecond, the bird
        descends in one second while not climbing.
    CLIMB_SPEED: With which speed, in pixels per millisecond, the bird
        ascends in one second while climbing, on average.  See also the
        Bird.update docstring.
    CLIMB_DURATION: The number of milliseconds it takes the bird to
        execute a complete climb.
    """
    # brain of our bird, is an MLP NN: an input layer with 4-5 units, one hidden layer with 4-5 units, and one output

    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.4
    CLIMB_DURATION = 150

    def think(self, pipes):  # bird decides weather it should jump or not
        cur_pipe = (pipes[0]) if pipes[0].x > 0 else pipes[-1]
        y_velocity = self.CLIMB_SPEED if self.msec_to_climb > 0 else self.SINK_SPEED  # the current velocity of bird
        # our inputs are based on 5 factors: the bird's y, the pipes top and bottom y, the coming pipes x,
        # the birds y_velocity
        inputs = [self.y / WIN_HEIGHT, cur_pipe.top_height_px / WIN_HEIGHT,
                  (WIN_HEIGHT - cur_pipe.bottom_height_px) / WIN_HEIGHT, cur_pipe.x / WIN_WIDTH, y_velocity/10]

        output = self.brain.predict(inputs)  # deciding weather bird should jump or not based on its brain input
        # print(output)
        if output[0][0] >= output[0][1]:  # just a check to see if we should climb or not
            self.msec_to_climb = Bird.CLIMB_DURATION * 1.5
        # print(f"{output} and {output[0][0]}")
        return output

    def __init__(self, x, y, msec_to_climb, images, brain=None):
        """Initialise a new Bird instance.

        Arguments:
        x: The bird's initial X coordinate.
        y: The bird's initial Y coordinate.
        msec_to_climb: The number of milliseconds left to climb, where a
            complete climb lasts Bird.CLIMB_DURATION milliseconds.  Use
            this if you want the bird to make a (small?) climb at the
            very beginning of the game.
        images: A tuple containing the images used by this bird.  It
            must contain the following images, in the following order:
                0. image of the bird with its wing pointing upward
                1. image of the bird with its wing pointing downward
        """
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self.brain = NeuralNetwork(5, 5, 2) if brain is None else brain
        self.alive = True
        self.score = 0
        self.fitness = 0

    def update(self, delta_frames=1):
        """Update the bird's position.

        This function uses the cosine function to achieve a smooth climb:
        In the first and last few frames, the bird climbs very little, in the
        middle of the climb, it climbs a lot.
        One complete climb lasts CLIMB_DURATION milliseconds, during which
        the bird ascends with an average speed of CLIMB_SPEED px/ms.
        This Bird's msec_to_climb attribute will automatically be
        decreased accordingly if it was > 0 when this method was called.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """''
        self.score += 1
        if self.msec_to_climb > 0 and 0 < self.y:
            frac_climb_done = 1 - self.msec_to_climb / Bird.CLIMB_DURATION
            self.y -= (Bird.CLIMB_SPEED * frames_to_msec(delta_frames) *
                       (1 - math.cos(frac_climb_done * math.pi)))
            self.msec_to_climb -= frames_to_msec(delta_frames)
        elif self.y < WIN_HEIGHT - Bird.HEIGHT:
            self.y += Bird.SINK_SPEED * frames_to_msec(delta_frames)

    @property
    def image(self):
        """Get a Surface containing this bird's image.

        This will decide whether to return an image where the bird's
        visible wing is pointing upward or where it is pointing downward
        based on pygame.time.get_ticks().  This will animate the flapping
        bird, even though pygame doesn't support animated GIFs.
        """
        return self._img_wingup

    @property
    def mask(self):
        """Get a bitmask for use in collision detection.

        The bitmask excludes all pixels in self.image with a
        transparency greater than 127."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        """Get the bird's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle.

    A PipePair has a top and a bottom pipe, and only between them can
    the bird pass -- if it collides with either part, the game is over.

    Attributes:
    x: The PipePair's X position.  This is a float, to make movement
        smoother.  Note that there is no y attribute, as it will only
        ever be 0.
    image: A pygame.Surface which can be blitted to the display surface
        to display the PipePair.
    mask: A bitmask which excludes all pixels in self.image with a
        transparency greater than 127.  This can be used for collision
        detection.
    top_pieces: The number of pieces, including the end piece, in the
        top pipe.
    bottom_pieces: The number of pieces, including the end piece, in
        the bottom pipe.

    Constants:
    WIDTH: The width, in pixels, of a pipe piece.  Because a pipe is
        only one piece wide, this is also the width of a PipePair's
        image.
    PIECE_HEIGHT: The height, in pixels, of a pipe piece.
    ADD_INTERVAL: The interval, in milliseconds, in between adding new
        pipes.
    """

    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False
        self.active = True
        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()  # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -  # fill window from top to bottom
             3 * Bird.HEIGHT -  # make room for bird to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT  # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces) - 1
        # here we can change width of pipes to change game difficulty
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces - 1

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i * PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        """Update the PipePair's position.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        global ANIMATION_SPEED, nm
        # if keyboard.is_pressed('+'):
        #     Bird.CLIMB_SPEED *= 2
        #     Bird.SINK_SPEED *= 2
        #     Bird.CLIMB_DURATION /= 2
        #     PipePair.ADD_INTERVAL /= 2
        #     ANIMATION_SPEED *= 2
        #     print('game speed *2')
        #     time.sleep(0.1)
        # if keyboard.is_pressed('-'):
        #     Bird.CLIMB_SPEED /= 2
        #     Bird.SINK_SPEED /= 2
        #     Bird.CLIMB_DURATION *= 2
        #     PipePair.ADD_INTERVAL *= 2
        #     ANIMATION_SPEED /= 2
        #     print('game speed /2')
        #     time.sleep(0.15)

        self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)

    def collides_with(self, bird):
        """Get whether the bird collides with a pipe in this PipePair.

        Arguments:
        bird: The Bird which should be tested for collision with this
            PipePair.
        """
        return pygame.sprite.collide_mask(self, bird)


def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    bird-wingup: An image of the bird with its wing pointing upward.
        Use this and bird-wingdown to create a flapping bird.
    bird-wingdown: An image of the bird with its wing pointing downward.
        Use this and bird-wingup to create a flapping bird.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folderWebGL
        (dirname(__file__)/images/). All images are converted before being
        returned to speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        # Look for images relative to this script, so we don't have to "cd" to
        # the script's directory before running it.
        # See also: https://github.com/TimoWilken/flappy-bird-pygame/pull/3
        file_name = os.path.join(os.path.dirname(__file__),
                                 'images', img_file_name)
        img = pygame.image.load(file_name)  # we want to get latest non-negative pipe

        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping bird -- animated GIFs are
            # not supported in pygame
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png')}


def frames_to_msec(frames, fps=FPS):
    """Convert frames to milliseconds at the specified framerate.

    Arguments:
    frames: How many frames to convert to milliseconds.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):
    """Convert milliseconds to frames at the specified framerate.

    Arguments:
    milliseconds: How many milliseconds to convert to frames.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return fps * milliseconds / 1000.0


def main():
    """The application's entry point.

    If someone executes this module (instead of importing it, for
    example), this function is called.
    """

    pygame.init()
    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Pygame Flappy Bird')

    clock = pygame.time.Clock()
    game_score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    highest_game_score_font = pygame.font.SysFont(None, 22, bold=True)
    images = load_images()

    birds = []  # we need more than one bird for our GA algorithm

    for i in range(TOTAL):
        birds.append((Bird(50, int(WIN_HEIGHT / 2 - Bird.HEIGHT / 2), 2,
                           (images['bird-wingup']))))

    pipes = deque()

    frame_clock = 0  # this counter is only incremented if the game isn't paused
    game_score = 0
    done = paused = False
    highest_game_score = 0
    cycles = cycles_copy = 1
    n = 1
    generation_num = 1
    alive_birds = TOTAL  # checker to see if all brids dead
    while not done:
        for n in range(cycles):
            r = random.uniform(0, 1)
            clock.tick(FPS)

            # Handle this 'manually'.  If we used pygame.time.set_timer(),
            # pipe addition would be messed up when paused.
            if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
                pp = PipePair(images['pipe-end'], images['pipe-body'])
                pipes.append(pp)
                
            if keyboard.is_pressed('+'):
                cycles += 50  # draw once every x+50 frames
                time.sleep(0.15)
            if keyboard.is_pressed('-'):
                cycles = 1  # reset drawing to 1
                time.sleep(0.15)
            while pipes and not pipes[0].visible:
                pipes.popleft()

            for p in pipes:
                p.update()

            for bird in birds:
                if not bird.alive: continue
                if game_score % FRAME_SKIPS == 0:  # no need to think on every frame, think once every few frames
                    # print(f"bird {step}: {birds[step].think(pipes)}")
                    bird.think(pipes)

                pipe_collision = any(p.collides_with(bird) for p in pipes if p.active)
                if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                    bird.alive = False
                    alive_birds -= 1
                    if alive_birds != 0:
                        continue
                    # if we reach here, ALL birds have died, lets bring next generation
                    if game_score > highest_game_score:
                        highest_game_score = game_score
                        # print(f"generation{generation_num} --> high_score!: {highest_game_score}")
                    generation_num += 1
                    game_score = 0
                    alive_birds = TOTAL
                    pipes[0].image.fill((0, 0, 0, 0))
                    pipes[0].active = False 
                    calculate_fitness(birds)
                    birds = next_generation(birds)
                    break

                bird.update()

            frame_clock += 1
            game_score += 1
        # drawing stuff
        game_score_surface = game_score_font.render(str(game_score), True, (255, 255, 255))
        game_score_x = WIN_WIDTH / 2 - game_score_surface.get_width() / 2
        display_surface.blit(game_score_surface, (game_score_x, PipePair.PIECE_HEIGHT))
        # update and display score
        # for p in pipes: # we can't
        #     if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
        #         score += 1
        #         p.score_counted = True
        gen_surface = game_score_font.render(f"GENERATION: {str(generation_num)}", True, (255, 0, 0))
        gen_x = WIN_WIDTH / 2 - gen_surface.get_width() / 2
        display_surface.blit(gen_surface, (gen_x, WIN_HEIGHT - 30))

        game_high_score_surface = highest_game_score_font.render(f"high score: {str(highest_game_score)}", True,
                                                                 (153, 50, 204))
        display_surface.blit(game_high_score_surface, (0, WIN_HEIGHT - game_high_score_surface.get_height()))

        pygame.display.flip()
        for x in (0, WIN_WIDTH / 2):
            display_surface.blit(images['background'], (x, 0))
        for p in pipes:
            display_surface.blit(p.image, p.rect)
        for bird in birds:
            if bird.alive:
                display_surface.blit(bird.image, bird.rect)

    print('Game over! Score: %i' % game_score)
    pygame.quit()

    # def draw():
    #     for i in range(cycles):


# ------------------ GA STUFF FROM HERE ON ------------------

def next_generation(birds):
    brds = []
    for i in range(TOTAL):
        brds.append(pick_a_bird(birds))
    return brds


def calculate_fitness(birds):
    total_bird_scores = 0
    for bird in birds:
        total_bird_scores += bird.score
        # print(total_scores)
    for idx, bird in enumerate(birds):
        bird.fitness = bird.score / total_bird_scores
        # print(f"bird[{idx}]: {bird.fitness}")


def pick_a_bird(birds):
    # here we want pick one of the dead birds (likelihood a bird is picked is based on its fitness)
    r = random.uniform(0, 1)
    idx = 0
    while r > 0:
        r -= birds[idx].fitness
        idx += 1
    idx -= 1
    bird = birds[idx]
    # we want to create a copy of the brain of the bird we picked, and put it into another bird
    child = Bird(50, int(WIN_HEIGHT / 2 - Bird.HEIGHT / 2), 2,
                 (load_images()['bird-wingup']), bird.brain.copy())
    # print(child.brain.model.get_weights())
    child.brain.mutate(0.1)

    return child


if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    main()
