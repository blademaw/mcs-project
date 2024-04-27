import numpy as np
import networkx as nx
from enum import Enum
from scipy import integrate
from tqdm import tqdm

from abc import ABC, abstractmethod
from typing import Callable, List, Any, Set


# STATS = {
#     "patch_ticks": 0,
#     "lambda_hj": [],
#     "lambda_v": [],
#     "patch1": [],
#     "num_infected": {
#         0: [],
#         1: [],
#         2: []
#     }
# }


class DiseaseState(Enum):
    SUSCEPTIBLE = 0
    EXPOSED     = 1
    INFECTED    = 2
    RECOVERED   = 3


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

        self.agents:  List[Agent] = np.full(num_agents, None, dtype=Agent)
        self.nodes:   List[Node]  = np.full(num_locations, None, dtype=Node)
        self.patches: List[Patch] = np.full(k, None, dtype=Patch)
        
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
            "r0": [],
            "num_movements": 0,
            "total_exposed": 0,
            "total_infected": 0,
            "total_recovered": 0,
            "total_time_in_state": [0, 0, 0, 0],
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
                 solve_timestep: float
                ):
        self.patch_id = patch_id

        # TODO: Figure out what the initial conditions are
        self.S, self.E, self.I = N0*(1-init_prop), 0, N0*init_prop
        # self.S, self.E, self.I = .995*N0, 0, .005*N0
        # self.S, self.E, self.I = np.random.random()*K_v/3, np.random.random()*K_v/3, np.random.random()*K_v/3
        # [ self.S, self.E, self.I ] = np.random.random(size=3)*K_v/3
        # self.S, self.E, self.I = np.random.random()*K_v/2, np.random.random()*K_v/2, 0
        # self.S, self.E, self.I = 0, 0, np.random.random()*K_v*0.005
        # self.S, self.E, self.I = np.random.random()*K_v, 0, 0

        # self.S, self.E, self.I = np.random.random()*K_v, 0, 0
        self.S, self.E, self.I = K_v, 0, 0


        # n = N0/3
        # self.S, self.E, self.I = n*(1-init_prop), 0, n*init_prop
        self.N_v = self.S + self.E + self.I
        self.K_v = K_v
        self.phi_v = phi_v
        self.r_v = r_v
        self.mu_v = mu_v
        self.nu_v = nu_v

        self.time = time
        self.timestep = timestep
        self.solve_timestep = solve_timestep

        self.lambda_v = None


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


class Agent:
    """A class representing an agent on a network."""
    def __init__(self,
                 state: DiseaseState,
                 node: int,
                 movement_rate: float,
                 movement_model: BaselineMovementModel,
                 nu_h: float,
                 mu_h: float,
                 model: Model) -> None:
        self.state = state
        self.node  = node
        self.model = model

        self.nu_h = nu_h
        self.mu_h = mu_h
        
        self.movement_rate  = movement_rate
        self.movement_model = movement_model

        self.num_ticks_in_state = 0


    def move(self) -> None:
        """Moves the agent randomly."""
        self.movement_model.move_agent(self, self.movement_rate)


    def update_state(self, lambda_hj: float) -> None:
        """Updates agent SEIR states stochastically."""
        self.num_ticks_in_state += 1

        r = np.random.random()
        match self.state:
            case DiseaseState.SUSCEPTIBLE:
                if r < 1 - np.exp(- self.model.timestep * lambda_hj):
                    self.state = DiseaseState.EXPOSED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_exposed"] += 1
            case DiseaseState.EXPOSED:
                self.model.statistics["total_time_in_state"][1] += 1
                if r < 1 - np.exp(- self.model.timestep * self.nu_h):
                    self.state = DiseaseState.INFECTED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_infected"] += 1
                    # NOTE: tracking
                    self.model.num_infected += 1
            case DiseaseState.INFECTED:
                self.model.statistics["total_time_in_state"][2] += 1
                if r < 1 - np.exp(- self.model.timestep * self.mu_h):
                    self.state = DiseaseState.RECOVERED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_recovered"] += 1
            case _:
                pass


class Activity:
    """Class to define an activity."""
    def __init__(self, activity_id: int, alpha: float) -> None:
        self.activity_id = activity_id
        
        assert alpha >= 0 and alpha <= 1, "Alpha must be in [0,1]"
        self.alpha = alpha


class Node:
    """A class representing a location (node) in the network model."""
    def __init__(self, node_id: int, activity: Activity, agents: Set[Agent]=None) -> None:
        self.node_id = node_id
        self.activity= activity
        
        self.agents: Set[Agent]  = set() if agents is None else agents
        
        
    def get_force_on_hosts(self, b_h: float, beta_hv: float, I_v: float, N_v: float) -> float:
        """Calculate the force of infection on hosts/agents for this node (lambda_{h,j})."""
        return (self.activity.alpha * b_h) * beta_hv * (I_v/N_v)
    
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the node."""
        self.agents.add(agent)
    
    
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent to the node."""
        self.agents.remove(agent)


class Patch:
    """Represents an area covering nodes with a corresponding mosquito 'cloud'."""
    def __init__(self,
                 k: int,
                 initial_infect_proportion: float,
                 density: float,
                 K_v: float,
                 sigma_v: float,
                 sigma_h: float,
                 phi_v: float,
                 beta_hv: float,
                 beta_vh: float,
                 nu_v: float,
                 mu_v: float,
                 r_v: float,
                 model: Model,
                 nodes: Set[Node]=None) -> None:
        self.k = k
        self.K_v = K_v
        self.sigma_v = sigma_v
        self.sigma_h = sigma_h
        self.model = model
        
        self.nodes: Set[Node] = set() if nodes is None else nodes
        
        self.mosquito_model = MosquitoModel(patch_id=k,
                                            N0=K_v,
                                            init_prop=initial_infect_proportion,
                                            density=density,
                                            K_v=K_v,
                                            phi_v=phi_v,
                                            r_v=r_v,
                                            mu_v=mu_v,
                                            nu_v=nu_v,
                                            time=model.time,
                                            timestep=model.timestep,
                                            solve_timestep=model.mosquito_timestep
                                            )

        # derived patch-specific values
        self.S_h = None
        self.E_h = None
        self.I_h = None
        self.R_h = None
        self.N_h = None

        self.S_hat_h = None
        self.E_hat_h = None
        self.I_hat_h = None
        self.R_hat_h = None
        self.N_hat_h = None

        self.beta_hv = beta_hv
        self.beta_vh = beta_vh

        self.b   = None
        self.b_v = None
        self.b_h = None


    # TODO: delete this.
    # def add_nodes(self, nodes_to_add: Iterable[int]) -> None:
    #     """Add nodes to the patch."""
    #     self.nodes = set.union(self.nodes, set(nodes_to_add))


    def tick(self) -> np.ndarray:
        """Advance the patch model by one time step."""
        self.model.statistics["patch_ticks"] += 1
        # Update patch values from ABM (agent statistics)
        self._update_patch_values()

        # Advance the patch model (EBM)
        lambda_v = self.get_force_on_vectors()
        self.model.statistics["lambda_v"].append(lambda_v)
        self.mosquito_model.tick(lambda_v)

        # Calculate effective reproduction rate
        if self.model.time == 0:
            m = self.mosquito_model
            r_hv = (m.nu_v/(m.mu_v+m.nu_v) * self.sigma_v/m.mu_v * self.beta_hv * self.sigma_h*self.N_hat_h/(self.sigma_h*self.N_hat_h + self.sigma_v*m.N_v))*((self.N_hat_h-self.I_hat_h)/self.N_hat_h)
            nu_h, mu_h = 1/5, 1/6
            r_vh = (nu_h/(mu_h+nu_h) * self.sigma_h/mu_h * self.sigma_v*m.N_v/(self.sigma_h*self.N_hat_h + self.sigma_v*m.N_v) * self.beta_vh)*(m.S/m.N_v)
            self.model.statistics["r0"].append(r_hv*r_vh)

        # For each node in this patch, progress disease states of agents
        lambda_hj=None
        for node in self.nodes:
            lambda_hj = node.get_force_on_hosts(b_h=self.b_h,
                                                beta_hv=self.beta_hv,
                                                I_v=self.mosquito_model.I,
                                                N_v=self.mosquito_model.N_v)
            self.model.statistics["lambda_hj"].append(lambda_hj)
            for agent in node.agents:
                agent.update_state(lambda_hj)

        # print(lambda_v, lambda_hj)


    def _update_patch_values(self) -> None:
        """Update the internal values of a patch."""
        seirs, seirs_hat = self._count_agents_in_patch()
        self.N_h, self.N_hat_h = seirs.sum(), seirs_hat.sum()
        
        [self.S_h, self.E_h, self.I_h, self.R_h] = seirs
        [self.S_hat_h, self.E_hat_h, self.I_hat_h, self.R_hat_h] = seirs_hat

        m = self.mosquito_model # for brevity
        
        self.b   = (self.sigma_v * m.N_v * self.sigma_h * self.N_hat_h)/(self.sigma_v * m.N_v + self.sigma_h * self.N_hat_h)
        self.b_v = self.b/m.N_v
        self.b_h = self.b/self.N_hat_h


    def _count_agents_in_patch(self):
        """Count agents in this patch."""
        seirs     = np.array([0, 0, 0, 0])
        seirs_hat = np.array([0, 0, 0, 0])

        for node in self.nodes:
            cur_seirs = np.array([0, 0, 0, 0])
            
            for agent in node.agents:
                match agent.state:
                    case DiseaseState.SUSCEPTIBLE:
                        cur_seirs[0] += 1
                    case DiseaseState.EXPOSED:
                        cur_seirs[1] += 1
                    case DiseaseState.INFECTED:
                        cur_seirs[2] += 1
                    case DiseaseState.RECOVERED:
                        cur_seirs[3] += 1

            seirs     += cur_seirs
            seirs_hat += node.activity.alpha * cur_seirs

        self.model.statistics["num_infected"][self.k].append(seirs[2])
        return seirs, seirs_hat


    def get_force_on_vectors(self) -> float:
        """Calculate the force of infection on vectors for this patch (lambda_v)."""
        return self.b_v * self.beta_vh * (self.I_hat_h/self.N_hat_h)

