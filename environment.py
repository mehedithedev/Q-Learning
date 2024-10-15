import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

# Setting up the random seed for reproducibility and some constants for grid size.
np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width

# This is the main environment class which inherits from Tkinter's Tk class.
class Env(tk.Tk):
    def __init__(self):
        # Initializing the environment and setting up the window title, size, and other essentials
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # Actions: up, down, left, right.
        self.n_actions = len(self.action_space)  # The number of possible actions.
        self.title("Let's play!") # Window title
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT)) # Setting up window size based on grid units.
        
        # Loading the game images (Mario, Bowser, etc.) and setting up the canvas.
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        # A list to store the text on the canvas, useful for displaying Q-values.
        self.texts = []
        # Stopwatch setup to display elapsed time in the corner of the window.
        self.stopwatch_label = tk.Label(self, text="00:00", font=("Helvetica", 24), bg='black', fg='red')
        self.stopwatch_label.place(x=WIDTH * UNIT - 100, y=10)
        self.start_time = time.time()  # Start counting time when the environment is created.
        self.update_stopwatch()
        

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='black', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        
        # Create grids lines for the game 
        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1, fill='gray', dash=(2, 3))  # Dotted horizontal lines) 
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            canvas.create_line(x0, y0, x1, y1, fill='gray', dash=(2, 3))  # Dotted horizontal lines

        # Create image
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0]) # Mario 
        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1]) # Bowser
        self.triangle2 = canvas.create_image(150, 250, image=self.shapes[2]) # Bullet Bill
        self.triangle3 = canvas.create_image(250, 350, image=self.shapes[3]) # Goomba 
        self.circle = canvas.create_image(250, 250, image=self.shapes[4]) # Peach

    # Pack all
        canvas.pack()

        return canvas
    
    # This function updates the stopwatch every second
    def update_stopwatch(self):
        elapsed_time = int(time.time() - self.start_time) # Calculate elapsed time
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        self.stopwatch_label.config(text=f"{minutes:02}:{seconds:02}")
        self.after(1000, self.update_stopwatch)# Call this function again in 1 second

    # This function loads the images for the game
    def load_images(self):
        rectangle = PhotoImage(
            Image.open("./img/Mario.png").resize((65, 65)))
        triangle1 = PhotoImage(
            Image.open("./img/Bowser.png").resize((65, 65)))
        triangle2 = PhotoImage(
            Image.open("./img/Bullet Bill.png").resize((65, 65)))
        triangle3 = PhotoImage(
            Image.open("./img/Goomba.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("./img/Peach.png").resize((65 , 65)))
        # Load the collision image
        collision_image = PhotoImage(
            Image.open("./img/Collision.png").resize((85, 85))) 
        goal_image = PhotoImage(
            Image.open("./img/heart.png").resize((85, 85)))  # Goal image

        return rectangle, triangle1, triangle2, triangle3, circle,  collision_image, goal_image

    # This method places text on the canvas to display the Q-values for a given state
    def text_value(self, row, col, contents, action, font='Helvetica', size=10,style='normal', anchor="nw"):
        
        if action == 0:
            origin_x, origin_y = 7, 42 # up
        elif action == 1:
            origin_x, origin_y = 85, 42 # down
        elif action == 2:
            origin_x, origin_y = 42, 5 # left
        else:
            origin_x, origin_y = 42, 77 # right

        # Calculate the exact coordinates based on the row and column
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        # Create the text on the canvas and add it to the list.
        text = self.canvas.create_text(x, y, fill="blue", text=contents,font=font, anchor=anchor)
        return self.texts.append(text)

    # This function clears and updates the entire grid with the Q-values from the Q-table
    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)  # Clear the previous values.
        self.texts.clear()  # Empty the list.
        # For each state in the grid, check if there's a Q-value, and print it
        for i in range(HEIGHT):
            for j in range(WIDTH):
                for action in range(0, 4):  # Loop through each action (up, down, left, right)
                    state = [i, j]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(j, i, round(temp, 2), action)

    # Convert the canvas coordinates to the corresponding grid state (row, column)
    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    # Convert the grid state (row, column) to canvas coordinates
    def state_to_coords(self, state):
        x = int(state[0] * 100 + 50)
        y = int(state[1] * 100 + 50)
        return [x, y]

    # Reset the environment by moving the player (Mario) back to the start position
    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y) # Move Mario back to the starting point
        # Reset the agent's image to the original
        self.canvas.itemconfig(self.rectangle, image=self.shapes[0])  # Reset to the original image
        self.render()
        # return observation
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle) # Return the new state after resetting
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            # Based on the action, adjust the movement vector to move up, down, left, or right
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

        # move agent
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # move rectangle to top level of canvas
        self.canvas.tag_raise(self.rectangle) # Ensure Mario stays on top of other objects
        
        next_state = self.canvas.coords(self.rectangle) # Get the new position after moving
    
        # Determine if the player has reached the goal (Peach) or collided with an enemy
        if next_state == self.canvas.coords(self.circle):
             # Change the agent's image upon reaching the goal
            self.canvas.itemconfig(self.rectangle, image=self.shapes[6])  # Goal image
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.triangle1),self.canvas.coords(self.triangle2),
        self.canvas.coords(self.triangle3)]:
             # Change the agent's image upon collision
            self.canvas.itemconfig(self.rectangle, image=self.shapes[5])  # Collision image
            reward = -100
            done = True
        else:
            reward = 0
            done = False
        
        next_state = self.coords_to_state(next_state)
        return next_state, reward, done
        
    def render(self):
        time.sleep(0.03)
        self.update()