from abc import ABC, abstractmethod
from typing import Any, Callable, List

import networkx as nx
import numpy as np
from scipy import integrate
from tqdm import tqdm


class Model(ABC):
    """Abstract class representing a custom instance of the hybrid ABM adapted from Manore et al."""
    def __init__(self,
                 time: float,
                 timestep: float,
                 mosquito_timestep: float) -> None:
        self.time = time
        self.timestep = timestep
        self.mosquito_timestep = mosquito_timestep
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def tick(self):
        pass


class BaselineModel(Model):
    """The baseline model from Manore et al."""
    def __init__(self,
                 k: int,
                 timestep: float,
                 movement_dist: Callable[None, float],
                 sigma_h_arr: np.ndarray,
                 sigma_v_arr: np.ndarray,
                 K_v_arr: np.ndarray,
                 patch_densities: np.ndarray,
                 phi_v_arr: np.ndarray,
                 beta_hv_arr: np.ndarray,
                 beta_vh_arr: np.ndarray,
                 nu_v_arr: np.ndarray,
                 mu_v_arr: np.ndarray,
                 r_v_arr: np.ndarray,
                 num_locations: int,
                 edge_prob: float,
                 num_agents: int,
                 initial_infect_proportion: float,
                 nu_h_dist: Callable[None, float],
                 mu_h_dist: Callable[None, float],
                 total_time: float,
                 mosquito_timestep: float) -> None:
        """
        Parameters
        ----------
        k : int
            Number of patches.
        timestep : float
            Time (in days) that one tick of the model represents, $\Delta t$.
        movement_dist : Callable[None, float]
            The distribution to draw agent $\rho$, the propensity to move nodes, from.
        sigma_h_arr : vector of size $k$
            $\sigma_h$ (the maximum number of mosquito bites an average host can sustain per time step) values for each patch.
        sigma_v_arr : vector of size $k$
            $\sigma_v$ (the number of times one mosquito would want to bite a host per time step if hosts were freely available) values for each patch.
        K_v_arr : vector of size $k$
            Carrying capacities for each patch.
        patch_densities : vector of size $k$
            Densities for each patch.
        phi_v_arr : vector of size $k$
            Emergence rates of mosquitoes for each patch.
        beta_hv_arr : vector of size $k$
            Probability of mosquito-to-host transmission for each patch.
        beta_vh_arr : vector of size $k$
            Probability of host-to-mosquito transmission for each patch.
        nu_v_arr : vector of size $k$
            Mosquito E->I rate for each patch.
        mu_v_arr : vector of size $k$
            Mosquito death rate per patch.
        num_locations : int
            Number of locations/nodes in the graph.
        edge_prob : float
            Probability of connecting two edges in the graph (p).
        num_agents : int
            Number of agents/hosts in the simulation.
        initial_infect_proportion : float
            Proportion of agents initially infected per patch.
        nu_h_dist : Callable[None, float]
            Distribution to draw $\nu_h$ from for each agent (E->I rate).
        mu_h_dist : Callable[None, float]
            Distribution to draw $\mu_h$ from for each agent (I->R rate).
        total_time : float
            Time (in days) to run simulation for.
        mosquito_timestep : float
            Not sure yet - suspect "RK" = Runge-Kutta.
        """
        # Assign model-specific parameters
        self.time = 0.0
        self.timestep = timestep
        self.total_time = total_time
        self.mosquito_timestep = mosquito_timestep
        self.movement_model = BaselineMovementModel(model=self)
        self.num_agents = num_agents

        self.agents:  List[Agent] = np.full(num_agents, None, dtype=Agent)
        self.nodes:   List[Node]  = np.full(num_locations, None, dtype=Node)
        self.patches: List[Patch] = np.full(k, None, dtype=Patch)

        self.K_v_arr = K_v_arr
        
        # NOTE: tracking
        self.num_infected = 0
        self.statistics = {
            "patch_ticks": 0,
            "lambda_hj": [],
            "lambda_v": [],
            "patch1": [],
            "num_infected": {
                0: [],
                1: [],
                2: []
            },
            "agent_disease_counts": [
                    [0] * int(self.total_time/self.timestep),
                    [0] * int(self.total_time/self.timestep),
                    [0] * int(self.total_time/self.timestep),
                    [0] * int(self.total_time/self.timestep)
                ],
            "r0": [],
            "num_movements": 0,
            "total_exposed": 0,
            "total_infected": 0,
            "total_recovered": 0,
            "total_time_in_state": [0, 0, 0, 0],
            "patch_sei": {0: np.zeros((int(total_time/timestep), 3)), 1: np.zeros((int(total_time/timestep), 3)), 2: np.zeros((int(total_time/timestep), 3))},
            "node_seir": {i: [] for i in range(num_locations)},
        }

        # Initialise network — Erdos-Renyi with n, p
        self.graph = nx.erdos_renyi_graph(num_locations, edge_prob)

        # Initialise nodes — distributed according to patch density
        node_patch_ids = np.random.choice(k, num_locations, p=patch_densities) # maps node id -> patch id
        activity = Activity(activity_id=0, alpha=1)                            # only one activity in baseline model
        
        for node_id in range(num_locations):
            node = Node(node_id=node_id, activity=activity)
            self.nodes[node_id] = node
            self.graph.nodes[node_id]["node"] = node

        # Initialise patches
        for patch_id in range(k):
            patch = Patch(
                k=patch_id,
                initial_infect_proportion=initial_infect_proportion,
                density=patch_densities[patch_id],
                K_v=K_v_arr[patch_id],
                sigma_v=sigma_v_arr[patch_id],
                sigma_h=sigma_h_arr[patch_id],
                phi_v=phi_v_arr[patch_id],
                beta_hv=beta_hv_arr[patch_id],
                beta_vh=beta_vh_arr[patch_id],
                nu_v=nu_v_arr[patch_id],
                mu_v=mu_v_arr[patch_id],
                r_v=r_v_arr[patch_id],
                model=self,
                nodes=self.nodes[np.where(node_patch_ids == patch_id)]
            )
            self.patches[patch_id] = patch


        # Initialise agents
        agent_disease_states = [DiseaseState.SUSCEPTIBLE] * num_agents
        for i in range(num_agents):
            agent = Agent(state=agent_disease_states[i],
                                   node=np.random.choice(num_locations), # chosen with equal probability
                                   movement_rate=movement_dist(),
                                   movement_model=self.movement_model,
                                   nu_h=nu_h_dist(),
                                   mu_h=mu_h_dist(),
                                   model=self
                                  )
            self.agents[i] = agent
            self.nodes[agent.node].add_agent(agent)

        # Initialise disease states _per patch_
        for patch in self.patches:
            for node in patch.nodes:
                for agent in node.agents:
                    agent.state = np.random.choice(
                        [DiseaseState.SUSCEPTIBLE, DiseaseState.INFECTED],
                        p=[1-initial_infect_proportion,
                           initial_infect_proportion])

        self.num_infected += np.array([agent.state==DiseaseState.INFECTED for agent in self.agents]).sum()
        self.statistics["total_infected"] += self.num_infected


    def tick(self) -> List[Any]:
        """Progress the model forward in time by the time step."""
        # (1) Update disease status of vectors and then hosts
        for patch in self.patches:
            patch.tick()
        
        # (2) Move agents randomly
        for agent in self.agents:
            agent.move()
            
        return []


    def run(self, with_progress=False):
        """Run the model until a certain number of time steps."""
        res = [] # TODO: refactor this into a pandas df where 1 row = tick, df = run
        
        ticks = int(self.total_time/self.timestep)
        
        if with_progress:
            for _ in tqdm(range(ticks)):
                res.append(self.tick())
                self.time += self.timestep
        else:
            for _ in range(ticks):
                res.append(self.tick())
                self.time += self.timestep
        # while self.time < self.total_time:
        
        # print(f"Patch ticks: {STATS['patch_ticks']}")
        return self.statistics, [self.num_infected]


class MosquitoModel:
    """Class representing a system of ODEs to be solved for a patch model."""
    def __init__(self,
                 patch_id: int,
                 N0: float,
                 init_prop: float,
                 density: float,
                 K_v: float,
                 phi_v: float,
                 r_v: float,
                 mu_v: float,
                 nu_v: float,
                 time: float,
                 timestep: float,
                 solve_timestep: float,
                 model: Model
                ):
        # TODO: verify & document the initial conditions.
        self.patch_id = patch_id
        # self.S, self.E, self.I = K_v/2 + np.random.random()*K_v/2, 0, 0
        # self.S, self.E, self.I = np.random.random()*K_v, 0, 0
        self.S, self.E, self.I = K_v, 0, 0

        self.N_v = K_v
        self.K_v = K_v
        self.phi_v = phi_v
        self.r_v = r_v
        self.mu_v = mu_v
        self.nu_v = nu_v

        self.time = time
        self.timestep = timestep
        self.solve_timestep = solve_timestep

        self.lambda_v = None
        self.model = model


    def tick(self, lambda_v):
        """Solve the mosquito model forward in time by the required amount."""
        self.lambda_v = lambda_v

        t   = np.arange(self.time,
                        self.time+self.timestep+self.solve_timestep,
                        self.solve_timestep)
        res = integrate.odeint(self._sei_rates,
                               (self.S, self.E, self.I),
                               t)
        self.N_v = res[-1].sum()
        self.S, self.E, self.I = res[-1].T

        self.model.statistics["patch_sei"][self.patch_id][int(self.model.time/self.model.timestep)] = res[-1].T


    def _sei_rates(self, X, t) -> np.ndarray:
        S, E, I = X
        N_v     = X.sum()

        h_v = (self.phi_v - self.r_v*N_v/self.K_v)*N_v
        
        dS = h_v - self.lambda_v * S - self.mu_v * S
        dE = self.lambda_v * S - self.nu_v * E - self.mu_v * E
        dI = self.nu_v * E - self.mu_v * I

        return np.array([dS, dE, dI])


class MovementModel(ABC):
    """Abstract class representing the logic to make an agent move on a network."""

    @abstractmethod
    def move_agent(self):
        pass


class BaselineMovementModel(MovementModel):
    """A baseline movement model as described in Manore et al."""
    def __init__(self, model: Model) -> None:
        self.model = model
        
        
    def move_agent(self, agent, rho: float) -> None:
        """Move an agent with probability 1 - e^{- delta t * rho}"""
        if np.random.random() < (1 - np.exp(-self.model.timestep*rho)):
            self.model.statistics["num_movements"] += 1

            # An agent moves to a connected node uniformly
            choices = self.model.graph.adj[agent.node]

            # If an agent can actually move
            if len(choices) > 0:
                new_node = np.random.choice(choices)
                
                self.model.graph.nodes[agent.node]["node"].remove_agent(agent)
                self.model.graph.nodes[new_node]["node"].add_agent(agent)
                
                agent.node = new_node


