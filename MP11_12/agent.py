import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def get_n(self, state, action):
        s = state
        return self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][action]
    
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        s = state
        self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][action] += 1


    def get_q(self, state, action):
        s = state
        return self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][action]
    

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 

        # lr = learning rate
        # ofv = optimal future value

        sp = s_prime
        Q_old = self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a]
        lr = (self.C / (self.C + self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a]))
        ofv = self.Q[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]][sp[5]][sp[6]][sp[7]].max()
        self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a] = Q_old + lr * (r + self.gamma * ofv - Q_old)


    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        if self._train:
            # Training

            # Check if game has not just started
            if self.s:
                # Game has been in progress

                # Determine reward
                reward = -0.1
                # Dies
                if dead:
                    reward = -1
                # Gets food
                elif points > self.points:
                    reward = 1
                    self.points = points
            
                # Update N table
                self.update_n(self.s, self.a)
                # Update Q table
                self.update_q(self.s, self.a, reward, s_prime)

            if dead:
                self.reset()
                return utils.RIGHT
                
            self.s = s_prime

            # Determine best action
            best_action = None # (Action, Q)

            # Check highest priority first: RIGHT > LEFT > DOWN > UP
            # Right

            if self.get_n(s_prime, utils.RIGHT) < self.Ne:
                self.a = utils.RIGHT
                return utils.RIGHT
            else:
                best_action = (utils.RIGHT, self.get_q(s_prime, utils.RIGHT))

            # Left
            if self.get_n(s_prime, utils.LEFT) < self.Ne:
                self.a = utils.LEFT
                return utils.LEFT
            elif best_action[1] < self.get_q(s_prime, utils.LEFT):
                best_action = (utils.LEFT, self.get_q(s_prime, utils.LEFT))

            # Down
            if self.get_n(s_prime, utils.DOWN) < self.Ne:
                self.a = utils.DOWN
                return utils.DOWN
            elif best_action[1] < self.get_q(s_prime, utils.DOWN):
                best_action = (utils.DOWN, self.get_q(s_prime, utils.DOWN))

            # Up
            if self.get_n(s_prime, utils.UP) < self.Ne:
                self.a = utils.UP
                return utils.UP
            elif best_action[1] < self.get_q(s_prime, utils.UP):
                best_action = (utils.UP, self.get_q(s_prime, utils.UP))

            self.a = best_action[0]
            return best_action[0]

        else:
            # Testing

            # Determine best action
            best_action = None # (Action, Q)

            # Check highest priority first: RIGHT > LEFT > DOWN > UP
            # Right
            best_action = (utils.RIGHT, self.get_q(s_prime, utils.RIGHT))

            # Left
            if best_action[1] < self.get_q(s_prime, utils.LEFT):
                best_action = (utils.LEFT, self.get_q(s_prime, utils.LEFT))

            # Down
            if best_action[1] < self.get_q(s_prime, utils.DOWN):
                best_action = (utils.DOWN, self.get_q(s_prime, utils.DOWN))

            # Up
            if best_action[1] < self.get_q(s_prime, utils.UP):
                best_action = (utils.UP, self.get_q(s_prime, utils.UP))

            return best_action[0]
    

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        
        # Environment Variables
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        rock_x = environment[5]
        rock_y = environment[6]

        # State Variables
        food_dir_x = 0
        food_dir_y = 0
        adjoining_wall_x = 0
        adjoining_wall_y = 0
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        
        # Food
        if food_x < snake_head_x:
            # Food on left of snake head
            food_dir_x = 1
        elif food_x > snake_head_x:
            # Food on right of snake head
            food_dir_x = 2
        else:
            # Food on same x coords as snake head
            food_dir_x = 0

        if food_y > snake_head_y:
            # Food on bottom of snake head
            food_dir_y = 2
        elif food_y < snake_head_y:
            # Food on top of snake head
            food_dir_y = 1
        else:
            # Food on same y coords as snake head
            food_dir_y = 0

        # Walls/Rocks
        if snake_head_x == 1 or (snake_head_x == (rock_x + 2) and snake_head_y == rock_y):
            # Wall/Rock is on snake head left
            adjoining_wall_x = 1
        elif snake_head_x == (self.display_width - 2) or (snake_head_x == (rock_x - 1) and snake_head_y == rock_y):
            # Wall/Rock is on snake head right
            adjoining_wall_x = 2
        else:
            # Wall/Rock is on neither left or right
            adjoining_wall_x = 0

        if snake_head_y == 1 or (snake_head_x in (rock_x, rock_x + 1) and snake_head_y == (rock_y + 1)):
            # Wall/Rock is on snake head top
            adjoining_wall_y = 1
        elif snake_head_y == (self.display_height - 2) or (snake_head_x in (rock_x, rock_x + 1) and snake_head_y == (rock_y - 1)):
            # Wall/Rock is on snake head bottom
            adjoining_wall_y = 2
        else:
            # Wall/Rock is on neither top or bottom
            adjoining_wall_y = 0

        # Body
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0
        
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1
        else:
            adjoining_body_bottom = 0
        
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0
        
        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
