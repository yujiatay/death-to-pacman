import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        self.collected = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Coins(Entity):
    def __init__(self):
        super(Coins, self).__init__()
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 1
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.discrete_action = True

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.coins

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # print(1,p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # print(2,p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b,p_force,a,b)
                # print(4,p_force)
                # if 'agent' in entity_a.name:
                #     # print(3,f_a)
                # if 'agent' in entity_b.name:
                    # print(4,f_b)
                #[f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                # if entity_a.movable: print("F_a",entity_a.name,f_a)
                # if entity_b.movable: print(entity_b.name,f_b)
                if(f_a is not None):
                    # if(p_force[a] is None): p_force[a] = 0.0
                    # print(5,p_force[a])
                    p_force[a] = f_a + p_force[a]
                    # print(6,p_force[a])

                if(f_b is not None):
                    # if(p_force[b] is None): p_force[b] = 0.0
                    # print(7,p_force[b])
                    p_force[b] = f_b + p_force[b]
                    # print(8,p_force[b])
                    # print(4, p_force)
                # if entity_a.movable: print("P_force",entity_a.name,p_force[a])
                # if entity_b.movable: print(entity_b.name,p_force[b])
        # print(p_force)
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            # print(10,p_force[i])
            # print(type(p_force[i]))
            if np.all(p_force[i] == 0 ):
                # print("this 2")
                # print(entity.state.p_vel)
                entity.state.p_vel *= -1.5
            elif (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                # print(7,entity.state.p_vel)
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            updated = False
            for ent in self.entities:
                if 'landmark' in ent.name:
                    new_pos = entity.state.p_pos + entity.state.p_vel * self.dt
                    next_delta_pos = new_pos - ent.state.p_pos
                    dist_min_hor = entity.hor + ent.hor
                    dist_min_vert = entity.vert + ent.vert
                    if abs(next_delta_pos[0]) <= dist_min_hor and abs(next_delta_pos[1])<=dist_min_vert :
                        entity.state.p_pos = entity.state.p_pos
                        updated = True
            if not updated:
                entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    # def get_collision_force(self, entity_a, entity_b):
    #     if (not entity_a.collide) or (not entity_b.collide):
    #         return [None, None] # not a collider
    #     if (entity_a is entity_b):
    #         return [None, None] # don't collide against itself
    #     # compute actual distance between entities
    #     delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
    #     dist = np.sqrt(np.sum(np.square(delta_pos)))
    #     # minimum allowable distance
    #     dist_min = entity_a.size + entity_b.size
    #     # softmax penetration
    #     k = self.contact_margin
    #     penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
    #     force = self.contact_force * delta_pos / dist * penetration
    #     force_a = +force if entity_a.movable else None
    #     force_b = -force if entity_b.movable else None
    #     return [force_a, force_b]

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b,p_force,a,b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if ('coin' in entity_a.name) or ('coin' in entity_b.name):
            # print("THISTHITSTIHSGDAFS")
            return [None,None]
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        entity_a_state_p_vel = entity_a.state.p_vel * (1 - self.damping) if entity_a.movable else 0.0
        entity_b_state_p_vel = entity_b.state.p_vel * (1 - self.damping) if entity_b.movable else 0.0
        entity_a_p_vel = entity_a_state_p_vel + (p_force[a] / entity_a.mass) * self.dt if entity_a.movable else 0.0
        entity_b_p_vel = entity_b_state_p_vel + (p_force[b] / entity_b.mass) * self.dt if entity_b.movable else 0.0

        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos

        # if entity_a_p_vel is None and entity_b_p_vel is None:
        #     next_delta_pos = (entity_a.state.p_pos) - (
        #             entity_b.state.p_pos)
        # elif entity_a_p_vel is None:
        #     next_delta_pos = (entity_a.state.p_pos) - (
        #                 entity_b.state.p_pos + entity_b_p_vel)
        # elif entity_b.state.p_vel is None:
        #     next_delta_pos = (entity_a.state.p_pos+ entity_a_p_vel) - (
        #             entity_b.state.p_pos)
        # else:
        #     next_delta_pos = (entity_a.state.p_pos + entity_a_p_vel) - (entity_b.state.p_pos + entity_b_p_vel)
        next_delta_pos = ((entity_a.state.p_pos + entity_a_p_vel)*self.dt) - ((entity_b.state.p_pos +
                                                                               entity_b_p_vel)*self.dt)
        # print(delta_pos)
        # dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance

        dist_min_hor = entity_a.hor + entity_b.hor
        dist_min_vert = entity_a.vert + entity_b.vert
        # softmax penetration
        # k = self.contact_margin
        # penetration = np.logaddexp(0, -(delta_pos[0] - dist_min_hor + delta_pos[1] - dist_min_vert)/k)*k
        force_a = np.zeros(2)
        force_b = np.zeros(2)

        if abs(delta_pos[0]) <= dist_min_hor and abs(delta_pos[1])<=dist_min_vert:
            # print("this")
            # force_a = -p_force[a] if entity_a.movable else None
            # force_b = -p_force[b] if entity_b.movable else None
            if abs(next_delta_pos[0])>abs(delta_pos[0]) and abs(next_delta_pos[1])>abs(delta_pos[1]):
                force_a = 0 if entity_a.movable else None
                force_b = 0 if entity_b.movable else None
            else:
                # print("this 1")
                force_a = -p_force[a] if entity_a.movable else None
                force_b = -p_force[b] if entity_b.movable else None


            # for i,dist in enumerate(next_delta_pos):
            #
            #     if abs(dist) > abs(delta_pos[i]):
            #         force_a[i] = 0 if entity_a.movable else None
            #         force_b[i] = 0 if entity_b.movable else None
            #     else:
            #         force_a[i] = -p_force[a][i] if entity_a.movable else None
            #         force_b[i] = -p_force[b][i] if entity_b.movable else None
            # force_a = -p_force[a] if entity_a.movable else None
            # force_b = -p_force[b] if entity_b.movable else None
        else:
            # force = self.contact_force * delta_pos / dist * penetration
            force_a = 0 if entity_a.movable else None
            force_b = 0 if entity_b.movable else None
        # force_a = +force if entity_a.movable else None
        # force_b = -force if entity_b.movable else None
        return [force_a, force_b]