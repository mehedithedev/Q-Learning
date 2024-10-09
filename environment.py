import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image, ImageSequence

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.fire_frames = self.load_fire_frames()  # Initialize fire_frames before _build_canvas
        self.canvas = self._build_canvas()
        self.texts = []
        self.current_fire_frame = 0
        self.animate_fire()

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1, fill='light grey')
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1, fill='light grey')

        # add img to canvas
        self.q = canvas.create_image(50, 50, image=self.shapes[0])
        self.fire1 = canvas.create_image(250, 150, image=self.fire_frames[0])
        self.fire2 = canvas.create_image(150, 250, image=self.fire_frames[0])
        self.fire3 = canvas.create_image(250, 350, image=self.fire_frames[0])
        self.todd = canvas.create_image(250, 250, image=self.shapes[2])

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = ImageTk.PhotoImage(
            Image.open("./img/rectangle.png").resize((65, 65)))
        fire = ImageTk.PhotoImage(
            Image.open("./img/fire.gif").resize((65, 65)))
        todd = ImageTk.PhotoImage(
            Image.open("./img/todd.png").resize((190, 100)))  # Adjust size as needed

        return [rectangle, fire, todd]

    def load_fire_frames(self):
        fire_image = Image.open("./img/fire.gif")
        frames = [ImageTk.PhotoImage(frame.copy().resize((65, 65))) for frame in ImageSequence.Iterator(fire_image)]
        return frames

    def animate_fire(self):
        self.current_fire_frame = (self.current_fire_frame + 1) % len(self.fire_frames)
        self.canvas.itemconfig(self.fire1, image=self.fire_frames[self.current_fire_frame])
        self.canvas.itemconfig(self.fire2, image=self.fire_frames[self.current_fire_frame])
        self.canvas.itemconfig(self.fire3, image=self.fire_frames[self.current_fire_frame])
        self.after(100, self.animate_fire)  # Adjust the delay as needed

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):

        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 7, 82
        elif action == 3:
            origin_x, origin_y = 85, 82

        self.texts.append(self.canvas.create_text(
            origin_x + (UNIT * col), origin_y + (UNIT * row),
            text=contents, font=(font, size, style), anchor=anchor))

    def step(self, action):
        state = self.canvas.coords(self.q)
        base_action = [0, 0]
        
        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # Move the agent
        self.canvas.move(self.q, base_action[0], base_action[1])
        self.update()

        # Next state
        next_state = self.canvas.coords(self.q)

        # Reward function
        if next_state == self.canvas.coords(self.todd):
            reward = 1
            done = True
        elif next_state in [self.canvas.coords(self.fire1), self.canvas.coords(self.fire2), self.canvas.coords(self.fire3)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return next_state, reward, done

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.q)
        self.q = self.canvas.create_image(50, 50, image=self.shapes[0])
        return self.canvas.coords(self.q)

    def render(self):
        time.sleep(0.03)
        self.update()


if __name__ == "__main__":
    env = Env()
    env.mainloop()