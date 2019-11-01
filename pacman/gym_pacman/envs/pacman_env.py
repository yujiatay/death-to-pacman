import copy
import heapq
import json
import math
import os

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .graphicsDisplay import PacmanGraphics
from .layout import getLayout, getRandomLayout
from .pacman import ClassicGameRules
from .pacmanAgents import OpenAIAgent

DEFAULT_GHOST_TYPE = 'DirectionalGhost'

PACMAN_ACTIONS = ['North', 'South', 'East', 'West', 'Stop']
pacman_actions_index = [0, 1, 2, 3, 4]

PACMAN_DIRECTIONS = ['North', 'South', 'East', 'West']
ROTATION_ANGLES = [0, 180, 90, 270]

fdir = '/'.join(os.path.split(__file__)[:-1])
print(fdir)
layout_params = json.load(open(fdir + '/../../layout_params.json'))

print("Layout parameters")
print("------------------")
for k in layout_params:
    print(k, ":", layout_params[k])
print("------------------")


class PacmanEnv(gym.Env):
    layouts = [
        'capsuleClassic', 'contestClassic', 'mediumClassic', 'mediumGrid', 'minimaxClassic', 'openClassic',
        'originalClassic', 'smallClassic', 'capsuleClassic', 'smallGrid', 'testClassic', 'trappedClassic',
        'trickyClassic'
    ]

    noGhost_layouts = [l + '_noGhosts' for l in layouts]

    MAX_MAZE_SIZE = (7, 7)
    num_envs = 1

    def __init__(self, want_display, numGhosts, MAX_EP_LENGTH, chosen_layout, obs_type, partial_obs_range, shared_obs,
                 timeStepObs, astarSearch, astarAlpha):
        # Newly added
        self.MAX_EP_LENGTH = MAX_EP_LENGTH
        self.obs_type = obs_type
        self.partial_obs_range = partial_obs_range
        self.shared_obs = shared_obs
        self.timeStepObs = timeStepObs
        self.astarSearch = astarSearch
        self.step_diff = None
        self.astarAlpha = astarAlpha
        self.chooseLayout(randomLayout=chosen_layout == "random", chosenLayout=chosen_layout)
        self.n = min(numGhosts + 1, self.layout.getNumGhosts() + 1)  # num of ghosts may be limited by map
        # set required vectorized gym env property
        self.prev_obs = [[] for i in range(self.n)]
        # self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        # # the above flag is used in step() as such:
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        self.want_display = want_display
        self.action_space = [spaces.Discrete(4) for i in range(self.n)]
        self.display = PacmanGraphics(1.0) if self.want_display else None
        # self._action_set = range(len(PACMAN_ACTIONS))
        self.location = None
        self.viewer = None
        self.done = False
        self.np_random = None

    def chooseLayout(self, randomLayout=True,
                     chosenLayout=None, no_ghosts=True):

        if randomLayout:
            self.layout = getRandomLayout(layout_params, self.np_random)
        else:
            if chosenLayout is None:
                if not no_ghosts:
                    chosenLayout = self.np_random.choice(self.layouts)
                else:
                    chosenLayout = self.np_random.choice(self.noGhost_layouts)
            self.chosen_layout = chosenLayout
            self.layout = getLayout(chosenLayout)
        self.maze_size = (self.layout.width, self.layout.height)

    def seed(self, seed=None):
        if self.np_random is None:
            self.np_random, seed = seeding.np_random(seed)
        # self.chooseLayout(randomLayout=True)
        self.chooseLayout(randomLayout=False, chosenLayout=self.chosen_layout)
        print(self.layout)
        return [seed]

    def reset(self, layout=None):
        self.step_counter = 0
        self.cum_reward = 0
        self.done = False

        # this agent is just a placeholder for graphics to work
        self.ghosts = [OpenAIAgent() for i in range(self.n - 1)]
        self.pacman = OpenAIAgent()
        self.agents = [self.pacman] + self.ghosts

        self.rules = ClassicGameRules(300)
        self.rules.quiet = True
        self.chooseLayout(randomLayout=False, chosenLayout=self.chosen_layout)
        # note that num of ghosts can be trimmed inside here based on the layout
        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts,
                                       self.display, quiet=True, catchExceptions=False)
        self.game.init()
        if self.want_display:
            self.display.initialize(self.game.state.data)
            self.display.updateView()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]

        self.location_history = [self.location]
        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history = [self.orientation]
        self.illegal_move_counter = 0

        obs_n = [self.observation(i, self.game.state.data.agentStates, self.game.state) for i in range(self.n)]

        if self.observation_space is None:  # first run initialization
            self.observation_space = []
            for i in range(self.n):
                self.observation_space.append(spaces.Box(low=0, high=max(self.layout.width, self.layout.height),
                                                shape=(len(obs_n[i]),),
                                                dtype=np.uint8))
            self.observation_space = np.array(self.observation_space)

        self.cum_reward = 0

        self.initial_info = {
            'past_loc': [self.location_history[-1]],
            'curr_loc': [self.location_history[-1]],
            'past_orientation': [[self.orientation_history[-1]]],
            'curr_orientation': [[self.orientation_history[-1]]],
            'illegal_move_counter': [self.illegal_move_counter],
            'ghost_positions': [self.ghostLocations],
            'step_counter': [[0]],
        }

        return obs_n

    def step(self, action_n):
        # implement code here to take an action
        if self.step_counter >= self.MAX_EP_LENGTH or self.done:
            self.step_counter += 1
            return np.zeros(self.observation_space), 0.0, True, {
                'past_loc': [self.location_history[-2]],
                'curr_loc': [self.location_history[-1]],
                'past_orientation': [[self.orientation_history[-2]]],
                'curr_orientation': [[self.orientation_history[-1]]],
                'illegal_move_counter': [self.illegal_move_counter],
                'step_counter': [[self.step_counter]],
                'ghost_positions': [self.ghostLocations],
                'r': [self.cum_reward],
                'l': [self.step_counter],
                # 'ghost_in_frame': [self.ghostInFrame],
                'episode': [{
                    'r': self.cum_reward,
                    'l': self.step_counter
                }]
            }

        agents_actions = []
        agent_illegal = []
        for i, action in enumerate(action_n):
            # print("action ndarray: ", action)
            legalMoves = self.game.state.getLegalActions(i)
            # print("legal moves: ", legalMoves)
            legalMoveIndexes = list(filter(lambda x: PACMAN_ACTIONS[x] in legalMoves, pacman_actions_index))
            # print("legal indexes: ", legalMoveIndexes)
            max_val = action[legalMoveIndexes[0]]
            best_move = legalMoveIndexes[0]  # do not move

            for j, act in enumerate(action):
                if act >= max_val:
                    original_best_move = j
                if j in legalMoveIndexes and act > max_val:
                    max_val = act
                    best_move = j
            # print("best move for index ", i, " is ", best_move)
            if original_best_move != best_move:
                agent_illegal.append(i)
            agents_actions.append(best_move)
        # print("agent_actions", agents_actions)
        agents_actions = [PACMAN_ACTIONS[i] for i in agents_actions]

        reward_n = self.game.step(agents_actions)

        if self.astarSearch:
            steps_to_pacman = self.call_search(self.game.state.data.agentStates, self.game.state)
            if self.step_diff == None:
                self.step_diff = steps_to_pacman
            else:
                for i, steps in enumerate(steps_to_pacman):
                    #print(steps - self.step_diff[i])
                    reward_n[i+1] -= steps - self.step_diff[i]
                self.step_diff = steps_to_pacman

        for i in agent_illegal:
            reward_n[i] -= 10

        # print(2,reward_n)
        # self.cum_reward += reward
        # # reward shaping for illegal actions
        # if illegal_action:
        #     reward -= 10

        done = self.game.state.isWin() or self.game.state.isLose()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.location_history.append(self.location)
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]

        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history.append(self.orientation)

        obs_n = [self.observation(i, self.game.state.data.agentStates, self.game.state) for i in range(self.n)]

        # extent = (self.location[0] - 1, self.location[1] - 1),(self.location[0] + 1, self.location[1] + 1),
        # self.ghostInFrame = any([ g[0] >= extent[0][0] and g[1] >= extent[0][1] and g[0] <= extent[1][0] and g[1] <= extent[1][1]
        #     for g in self.ghostLocations])
        self.step_counter += 1
        info = {
            'past_loc': [self.location_history[-2]],
            'curr_loc': [self.location_history[-1]],
            'past_orientation': [[self.orientation_history[-2]]],
            'curr_orientation': [[self.orientation_history[-1]]],
            'illegal_move_counter': [self.illegal_move_counter],
            'step_counter': [[self.step_counter]],
            'episode': [None],
            'ghost_positions': [self.ghostLocations],
            # 'ghost_in_frame': [self.ghostInFrame],
        }

        if self.step_counter >= self.MAX_EP_LENGTH:
            done = True

        self.done = done

        if self.done:  # only if done, send 'episode' info
            info['episode'] = [{
                'r': self.cum_reward,
                'l': self.step_counter
            }]
        return obs_n, reward_n, done, info, self.game.state.isWin() ,self.game.state.isLose()

    def observation(self, agent_index, agent_states, game_states):
        comm = []
        other_pos = []
        other_vel = []
        scared_array = []
        agent = agent_states[agent_index]

        for i, other in enumerate(agent_states):
            if i == agent_index:
                other_vel.append(other.getDirection())
                continue
            other_pos.append(np.array(other.getPosition()))
            other_vel.append(other.getDirection())

        for i, other in enumerate(agent_states):
            if i == 0:
                continue
            else:
                scared_array.append(int(other.scaredTimer > 0))
        # print(scared_array)

        for i in range(len(other_vel)):
            other_vel[i] = PACMAN_ACTIONS.index(other_vel[i])

        one_hot_vel = []
        for i in other_vel:
            one_hot = [0]*(5)
            one_hot[i] = 1
            one_hot_vel.extend(one_hot)
        one_hot_vel = np.array(one_hot_vel)

        if self.astarSearch:
            steps_to_pacman = self.call_search(self.game.state.data.agentStates, self.game.state)
            #  print(1, steps_to_pacman)




        # if agent_index == 0: print(other_vel)
        # print(PACMAN_ACTIONS.index(other_vel[0]))

        if self.obs_type == 'full_obs':
            capsule_loc = np.asarray(
                list(map(int, str(game_states.getCapsules_TF()).replace("T", "1").replace("F", "0").replace("\n", ""))))
            food_loc = np.asarray(
                list(map(int, str(game_states.getFood()).replace("T", "1").replace("F", "0").replace("\n", ""))))
            wall_loc = np.asarray(
                list(map(int, str(game_states.getWalls()).replace("T", "1").replace("F", "0").replace("\n", ""))))
            # return np.concatenate([agent.getDirection()] + [agent.getPosition()] + other_pos + other_vel)

            all_agent_grid = []
            for i in range(len(agent_states)):
                agent_grid = list(map(int,str(game_states.getAgent_grid(i)).replace("T", "1").replace("F","0").replace("\n", "")))
                all_agent_grid.extend(agent_grid)
            all_agent_grid = np.array(all_agent_grid)


            if self.shared_obs:
                tmp = np.concatenate((all_agent_grid,
                                      one_hot_vel, capsule_loc,
                                      food_loc, wall_loc,
                                      scared_array))
                if self.astarSearch: tmp = np.concatenate((tmp,steps_to_pacman))
                if self.prev_obs[agent_index] == []:
                    self.prev_obs[agent_index] = np.zeros(len(tmp))

                if self.timeStepObs:
                    obs = np.concatenate((self.prev_obs[agent_index], tmp))
                    self.prev_obs[agent_index] = tmp
                else:
                    obs = tmp
            else:
                if agent_index == 0:
                    tmp = np.concatenate((all_agent_grid,one_hot_vel,
                                          capsule_loc, food_loc,
                                          wall_loc, scared_array))
                    if self.astarSearch: tmp = np.concatenate((tmp, steps_to_pacman))
                    if self.prev_obs[agent_index] == []:
                        self.prev_obs[agent_index] = np.zeros(len(tmp))
                    if self.timeStepObs:
                        obs = np.concatenate((self.prev_obs[agent_index], tmp))
                        self.prev_obs[agent_index] = tmp
                    else:
                        obs = tmp
                else:
                    tmp = np.concatenate((all_agent_grid,one_hot_vel,
                                          wall_loc, scared_array))
                    if self.astarSearch: tmp = np.concatenate((tmp, steps_to_pacman))
                    if self.prev_obs[agent_index] == []:
                        self.prev_obs[agent_index] = np.zeros(len(tmp))
                    if self.timeStepObs:
                        obs = np.concatenate((self.prev_obs[agent_index], tmp))
                        self.prev_obs[agent_index] = tmp
                    else:
                        obs = tmp
            return obs

        elif self.obs_type == 'partial_obs':
            partial_size = self.partial_obs_range
            part_wall = []
            part_food = []
            part_capsule = []
            numAgent = len(agent_states)
            width, height = game_states.getWidth(), game_states.getHeight()

            wall = game_states.getWalls()
            food = game_states.getFood()
            capsule = game_states.getCapsules_TF()

            all_agent_grid = []
            for i in range(numAgent):
                agent_grid = game_states.getAgent_grid(i)
                all_agent_grid.append(agent_grid)

            x, y = agent.getPosition()[0], agent.getPosition()[1]
            diff = (partial_size - 3) // 2

            partial_all_agent=[[]for i in range(numAgent)]

            for i in range(1 + diff, -2 - diff, -1):
                for j in range(-1 - diff, 2 + diff):
                    if y + i <= 0 or y + i >= height or x + j <= 0 or x + j >= width:
                        part_wall.append(1)
                        part_food.append(0)
                        part_capsule.append(0)
                        for k in range(numAgent):
                            partial_all_agent[k].append(0)
                    else:
                        part_wall.append(int(wall[int(x + j)][int(y + i)]))
                        part_food.append(int(food[int(x + j)][int(y + i)]))
                        part_capsule.append(int(capsule[int(x + j)][int(y + i)]))
                        for k in range(numAgent):
                            tmp = all_agent_grid[k]
                            partial_all_agent[k].append(int(tmp[int(x + j)][int(y + i)]))

            partial_all_agent =np.concatenate(partial_all_agent)

            if self.shared_obs:
                obs = np.concatenate((partial_all_agent, one_hot_vel, part_capsule,
                                      part_food, part_wall, scared_array))
                if self.astarSearch: obs = np.concatenate((obs, steps_to_pacman))
            else:
                if agent_index == 0:
                    obs = np.concatenate((partial_all_agent, one_hot_vel,
                                          part_capsule, part_food, part_wall, scared_array))
                    if self.astarSearch: obs = np.concatenate((obs, steps_to_pacman))
                else:
                    obs = np.concatenate((partial_all_agent, one_hot_vel, part_wall, scared_array))
                    if self.astarSearch: obs = np.concatenate((obs, steps_to_pacman))
            return obs

    # def get_action_meanings(self):
    #     return [PACMAN_ACTIONS[i] for i in self._action_set]

    # just change the get image function
    def _get_image(self):
        # get x, y
        if self.want_display:
            image = self.display.image
        w, h = image.size
        DEFAULT_GRID_SIZE_X, DEFAULT_GRID_SIZE_Y = w / float(self.layout.width), h / float(self.layout.height)

        extent = [
            DEFAULT_GRID_SIZE_X * (self.location[0] - 1),
            DEFAULT_GRID_SIZE_Y * (self.layout.height - (self.location[1] + 2.2)),
            DEFAULT_GRID_SIZE_X * (self.location[0] + 2),
            DEFAULT_GRID_SIZE_Y * (self.layout.height - (self.location[1] - 1.2))]
        extent = tuple([int(e) for e in extent])

        # self.image_sz = (84,84)
        self.image_sz = (500, 500)

        # image = image.crop(extent).resize(self.image_sz)
        image = image.resize(self.image_sz)
        return np.array(image)

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        # TODO: implement code here to do closing stuff
        if self.viewer is not None:
            self.viewer.close()
        if self.want_display:
            self.display.finish()

    def __del__(self):
        self.close()

    def call_search(self, agents, game_state):
        walls = list(map(list, str(game_state.getWalls()).split('\n')))
        walls.reverse()
        # walls = list(map(lambda x: list(x.replace('T', '1')), walls))
        pacman_pos = agents[0].getPosition()
        walls[pacman_pos[1]][pacman_pos[0]] = 'P'
        # print(1,walls[6])
        # print(2,walls[4])
        # print(2, pacman_pos)
        ghost_pos = list(map(lambda x: x.getPosition(), agents[1:]))
        puzzles = []
        for i in range(len(ghost_pos)):
            # print(len(walls))
            new_puzzle = copy.deepcopy(walls)
            # print(len(new_puzzle))
            col, row = ghost_pos[i]
            # print(row,col)
            new_puzzle[int(row)][int(col)] = 'G'
            for j in range(len(ghost_pos)):
                if ghost_pos[i] != ghost_pos[j]:
                    col2, row2 = ghost_pos[j]
                    new_puzzle[int(row2)][int(col2)] = 'T'
            puzzles.append(new_puzzle)
        distances = []
        for k in puzzles:
            search_state = Search(k)
            distances.append(search_state.solve())
        return distances


class PQ:
    def __init__(self):
        self.minheap = list()

    def push(self, value, entry_no, node):
        heapq.heappush(self.minheap, (value, entry_no, node))

    def pop(self):
        node = heapq.heappop(self.minheap)
        return node

    def is_empty(self):
        return len(self.minheap) == 0

    def get_print(self):
        return self.minheap


class Node:
    def __init__(self, moves, pos):
        self.moves = moves
        self.pos = pos

    def get_moves(self):
        return self.moves

    def get_pos(self):
        return self.pos


class Search(object):
    def __init__(self, init_state):
        self.state = init_state
        self.pacman_pos = None
        self.ghost_pos = None
        self.distances = []
        self.visited = [[False for j in range(len(self.state[i]))] for i in range(len(self.state))]
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 'T':
                    self.visited[i][j] = True
                if self.state[i][j] == 'G':
                    self.ghost_pos = (i, j)
                if self.state[i][j] == 'P':
                    self.pacman_pos = (i, j)

    def solve(self):
        pq = PQ()
        start = Node(0, self.get_ghost_position())
        pq.push(self.evaluation(start), 0, start)
        counter = 1
        if self.straight_line_dist(start) == 0:
            return 0
        else:
            while not pq.is_empty():
                priority, num, node = pq.pop()
                row, col = node.get_pos()
                if self.goal_test(node):
                    return node.get_moves()
                if self.visited[row][col] == True:
                    continue
                self.visited[row][col] = True
                for successor in self.get_successors(node):
                    s_row, s_col = successor.get_pos()
                    if self.visited[s_row][s_col] == False:
                        pq.push(self.evaluation(successor), counter, successor)
                        counter += 1
        return -1

    def get_pacman_position(self):
        return self.pacman_pos

    def get_ghost_position(self):
        return self.ghost_pos

    def get_grid_size(self):
        return (len(self.get_init_state()), len(self.get_init_state()[0]))

    def get_init_state(self):
        return self.state

    def goal_test(self, node):
        g_row, g_col = node.get_pos()
        p_row, p_col = self.get_pacman_position()
        return (g_row == p_row) and (g_col == p_col)

    def straight_line_dist(self, node):
        pacman = self.get_pacman_position()
        ghost = node.get_pos()
        if pacman == None or ghost == None:
            return 0
        total = (pacman[0] - ghost[0]) ** 2 + (pacman[1] - ghost[1]) ** 2
        return round(math.sqrt(total), 2)

    def evaluation(self, node):
        return node.get_moves() + self.straight_line_dist(node)

    def get_successors(self, node):
        successor_nodes = list()
        row, col = node.get_pos()
        grid_size = self.get_grid_size()
        number_of_moves = node.get_moves()
        if row - 1 >= 0:
            successor_nodes.append(Node(number_of_moves + 1, (row - 1, col)))
        if row + 1 < grid_size[0]:
            successor_nodes.append(Node(number_of_moves + 1, (row + 1, col)))
        if col - 1 >= 0:
            successor_nodes.append(Node(number_of_moves + 1, (row, col - 1)))
        if col + 1 < grid_size[1]:
            successor_nodes.append(Node(number_of_moves + 1, (row, col + 1)))
        return successor_nodes
