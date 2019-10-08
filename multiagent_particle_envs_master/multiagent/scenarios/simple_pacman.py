import numpy as np
from multiagent.core import World, Agent, Landmark,Coins
from multiagent.scenario import BaseScenario
#from pacman_layout import get_layout


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.complete = False
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        # num_landmarks = 3
        #world.landmark_ind = [[0.5,0.5],[0.5,0.7],[0.5,0.3]]
        #layout,num_landmarks = get_layout()
        #world.landmark_ind = np.array(layout)
        # layout, num_landmarks = [[0,0]],0
        #world.landmark_ind = np.array(layout)
        world.landmark_ind = np.array([[0,1,1,0.1],[0,-1,1,0.1],[1,0,0.1,1],[-1,0,0.1,1],[-0.4,0.4,0.2,0.2],
                                       [-0.5,-0.1,0.1,0.5],[0.3,0.2,0.3,0.4],[0.1,-0.3,0.3,0.3],[0.7,-0.6,0.1,0.2]])
        num_landmarks = len(world.landmark_ind)
        # world.agent_ind = np.array([[-0.7000000000000001, 0.8],[0.7000000000000001, 0.8],[-0.09999999999999995, -0.6000000000000001]])
        #world.agent_ind = np.array([[-0.5774193548387098, 0.935483870967742],[0.067741935483871,
        # 0.8709677419354839],[0.132258064516129, -0.8709677419354838]]) #for 31 x 31

        world.agent_ind = np.array([[0.8,0.7],[-0.6,0.7],[0.4,-0.8],[0.6,0.7],[-0.8,0],[-0.5,-0.8],[-0.1,0.1],[-0.1,0.4],[-0.3,-0.4]])
        #world.agent_ind = np.array([[0.8, 0.7], [-0.6, 0.7], [-0.4, -0.7]])

        # world.agent_ind = np.array([[0.7000000000000001, 0.8],[-0.09999999999999995, -0.6000000000000001]])
        # add agents
        # agent_rand_pos = []
        # agent_rand_pos.append(np.random.randint(0, 8))
        # agent_rand_pos.append(np.random.randint(0, 8))
        # while agent_rand_pos[1] == agent_rand_pos[0]:
        #     agent_rand_pos[1] = np.random.randint(0, 8)
        # agent_rand_pos.append(np.random.randint(0, 8))
        # while agent_rand_pos[2] == agent_rand_pos[0] or agent_rand_pos[2] == agent_rand_pos[1]:
        #     agent_rand_pos[2] = np.random.randint(0, 8)

        world.agents = [Agent() for i in range(num_agents)]

        world.coins_ind = []
        for i in range(11):
            for j in range(11):
                world.coins_ind.extend(
                    [[i / 10, j / 10, 0.01, 0.01], [-i / 10, j / 10, 0.01, 0.01], [i / 10, -j / 10, 0.01, 0.01],
                     [-i / 10, -j / 10, 0.01, 0.01]])
        world.coins_ind = np.array(world.coins_ind)
        temp = []
        for coin in world.coins_ind:
            add = True
            for landmark in world.landmark_ind:
                delta_pos = coin[:2] - landmark[:2]
                dist_min_hor = coin[2] + landmark[2]
                dist_min_vert = coin[3] + landmark[3]
                if (abs(delta_pos[0]) <= dist_min_hor and abs(delta_pos[1]) <= dist_min_vert):
                    add = False
                    break
            if add:
                temp.append(coin)
        world.coins_ind = np.array(temp)
        num_coins = len(world.coins_ind)

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.05 if agent.adversary else 0.05
            agent.hor = 0.05 if agent.adversary else 0.05
            agent.vert = 0.05 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 5.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.5
            scale = -15
            if agent.adversary:
                agent.shape =  np.array([( 0,0.3/scale ),( 0.25/scale, 0.75/scale ),( 0.5/scale,  0.3 /scale), (0.75/scale, 0.75/scale ),
                                         ( 0.75/scale, -0.5/scale ),( 0.5/scale,
                                                                                                        -0.75 /scale),
                                         (-0.5/scale,  -0.75/scale ),(-0.75/scale, -0.5/scale ),(-0.75/scale, 0.75/scale ),(-0.5/scale,
                                                                                                          0.3/scale ),
                                         (-0.25/scale, 0.75/scale )])
                # for i in agent.shape:
                #     i[1] = - i[1]

                # agent.lefteye = []
                # agent.righteye =[]
                # agent.lefteye
                # agent.leftpupil = []
                # agent.rightpupil = []
                # for i in range(res):
                #     ang = 2 * math.pi * i / res
                #     agent.lefteye.append((math.cos(ang) * radius -0.003, math.sin(ang) * radius + 0.003))
                #     agent.righteye.append((math.cos(ang) * radius + 0.003, math.sin(ang) * radius + 0.003))



            # agent_rand_pos =[]
            # agent_rand_pos.append(np.random.randint(0,8))
            # agent_rand_pos.append(np.random.randint(0, 8))
            # while agent_rand_pos[1]==agent_rand_pos[0]:
            #     agent_rand_pos[1] = np.random.randint(0, 8)
            # agent_rand_pos.append(np.random.randint(0, 8))
            # while agent_rand_pos[2]==agent_rand_pos[0] or agent_rand_pos[2]==agent_rand_pos[1]:
            #     agent_rand_pos[2] = np.random.randint(0, 8)

            agent.state.p_pos = np.array(world.agent_ind[i])
            agent.boundary = [agent.state.p_pos[1] + agent.vert, agent.state.p_pos[1] - agent.vert,
                                 agent.state.p_pos[0] - agent.hor, agent.state.p_pos[
                                     0] + agent.hor]
            agent.forRender = [[agent.hor, agent.vert],[-agent.hor,+agent.vert],[-agent.hor,-agent.vert],[+agent.hor,-agent.vert]]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False    
            landmark.size = 0.03
            landmark.boundary = False
            landmark.state.p_pos = np.array(world.landmark_ind[i][:2])
            landmark.hor = world.landmark_ind[i][2]
            landmark.vert = world.landmark_ind[i][3]
            landmark.boundary = [landmark.state.p_pos[1]+landmark.vert, landmark.state.p_pos[1]-landmark.vert,landmark.state.p_pos[0]-landmark.hor,landmark.state.p_pos[
                0]+landmark.hor]
            landmark.forRender = [[landmark.hor, landmark.vert],[-landmark.hor,+landmark.vert],[-landmark.hor,-landmark.vert],[+landmark.hor,-landmark.vert]]
        world.coins = [Coins() for i in range(num_coins)]
        for i, coin in enumerate(world.coins):
            coin.name = 'coin %d' % i
            coin.collide = True
            coin.movable = False
            coin.size = 0.02
            coin.boundary = False
            coin.state.p_pos = np.array(world.coins_ind[i][:2])
            coin.hor = world.coins_ind[i][2]
            coin.vert = world.coins_ind[i][3]
            coin.boundary = [coin.state.p_pos[1] + coin.vert, landmark.state.p_pos[1] - coin.vert,
                             coin.state.p_pos[0] - coin.hor, coin.state.p_pos[
                                 0] + landmark.hor]
            coin.forRender = [[coin.hor, coin.vert], [-coin.hor, +coin.vert], [-coin.hor, -coin.vert],
                              [+coin.hor, -coin.vert]]
            coin.state.collected = False
            
        # make initial conditions
        self.reset_world(world)
        return world

    # def reset_world(self, world):
    #     # random properties for agents
    #     for i, agent in enumerate(world.agents):
    #         agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
    #         # random properties for landmarks
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.color = np.array([0.25, 0.25, 0.25])
    #     # set random initial states
    #     for i,agent in enumerate(world.agents):
    #
    #         agent.state.p_pos = np.array(world.agent_ind[i])
    #         agent.state.p_vel = np.zeros(world.dim_p)
    #         agent.state.c = np.zeros(world.dim_c)
    #     for i, landmark in enumerate(world.landmarks):
    #         if not landmark.boundary:
    #             landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
    #             landmark.state.p_pos = np.array(world.landmark_ind[i])
    #             landmark.state.p_vel = np.zeros(world.dim_p)
    def reset_world(self, world):
        # print("############################################################################")
        # random properties for agents
        world.complete = False
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, coin in enumerate(world.coins):
            coin.color = np.array([1.0, 0.9, 0.0])
        # set random initial states
        agent_rand_pos =[]
        agent_rand_pos.append(np.random.randint(0,8))
        agent_rand_pos.append(np.random.randint(0, 8))
        while agent_rand_pos[1]==agent_rand_pos[0]:
            agent_rand_pos[1] = np.random.randint(0, 8)
        agent_rand_pos.append(np.random.randint(0, 8))
        while agent_rand_pos[2]==agent_rand_pos[0] or agent_rand_pos[2]==agent_rand_pos[1]:
            agent_rand_pos[2] = np.random.randint(0, 8)

        for i,agent in enumerate(world.agents):
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

            agent.state.p_pos = np.array(world.agent_ind[agent_rand_pos[i]])

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_pos = np.array(world.landmark_ind[i])
                landmark.state.p_vel = np.zeros(world.dim_p)
        for i, coin in enumerate(world.coins):
            coin.state.p_pos = np.array(world.coins_ind[i][:2])
            coin.state.p_vel = np.zeros(world.dim_p)
            coin.state.collected = False


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, entity_a, entity_b):
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist_min_hor = entity_a.hor + entity_b.hor
        dist_min_vert = entity_a.vert + entity_b.vert
        if abs(delta_pos[0]) <= dist_min_hor and abs(delta_pos[1])<=dist_min_vert:

            return True
        else:
            return False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    def entities(self,world):
        return [entity for entity in world.entities ]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward
    def doned(self,agent,world):
        return world.complete
    # def agent_reward(self, agent, world):
    #     # Agents are negatively rewarded if caught by adversaries
    #     rew = 0
    #     shape = True
    #     adversaries = self.adversaries(world)
    #     # if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
    #     #     for adv in adversaries:
    #     #         rew += 0.1 *10* np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))) #We can change this to
    #     #         # collecting pellets for rewards
    #     # if agent.collide:
    #     #     for a in self.adversaries(world):
    #     #         if self.is_collision(a, agent):
    #     #             rew -= 1000
    #     #         if self.is_collision(a,agent) and a in self.adversaries(world):
    #     #             # print("HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    #     #             world.complete = True
    #
    #     # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    #     # def bound(x):
    #     #     if x < 0.9:
    #     #         return 0
    #     #     if x < 1.0:
    #     #         return (x - 0.9) * 10
    #     #     return min(np.exp(2 * x - 2), 10)
    #     # for p in range(world.dim_p):
    #     #     x = abs(agent.state.p_pos[p])
    #     #     rew -= bound(x)
    #
    #     return rew

    def agent_reward(self, agent, world):
        rew = 0
        coin_lst = np.array([coin for coin in world.coins])
        if np.all(coin_lst == True):
            world.complete = True
            rew += 1000
        if agent.collide:
            for a in self.adversaries(world):
                if self.is_collision(a, agent):
                    rew -= 1000
                if self.is_collision(a,agent) and a in self.adversaries(world):
                    # print("HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                    world.complete = True
        for coin in self.entities(world):
            if "coin" in coin.name and coin.state.collected == False:
                # print("1")
                if self.is_collision(coin, agent):
                    # print('22222222222222222222222222222222222222222222222222222222222')
                    rew += 10
                    coin.state.collected = True
                    coin.color = np.array([1.0, 1.0, 1.0])

        # print(rew)
        return rew
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 *10* min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 1000
        # if agent.collide:
        #     for a in self.entities(world):
        #         if a in agents: continue
        #         if self.is_collision(a, agent):
        #             rew -= 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos_x = []
        entity_pos_y = []
        entity_size = []
        #Should add if the coins is collected anot.
        for entity in world.entities:
            #if not entity.boundary and not entity.movable:
            if not entity.movable:
                #entity_pos.append(entity.state.p_pos - agent.state.p_pos) #Need to include the size of the rectangle
                # also
                entity_pos_x.append([entity.state.p_pos[0] - agent.state.p_pos[0]])
                entity_pos_y.append([entity.state.p_pos[1] - agent.state.p_pos[1]])
                entity_size.append([entity.hor,entity.vert])
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)


        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos_x + entity_pos_y + entity_size+
                              other_pos +  other_vel)
