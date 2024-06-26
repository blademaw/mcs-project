from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from .agent import Activity, Agent, Node
from .patch import Patch, DiseaseState


class Model(ABC):
    """
    Abstract class representing a custom instance of the hybrid ABM adapted
    from Manore et al.

    Attributes
    ---
    time : float
        The current time of the model.

    timestep : float
        The amount between each time step.

    mosquito_timestep : float
        Amount to solve the mosquito ODE forward in.
    """
    def __init__(self,
                 time: float,
                 timestep: float,
                 mosquito_timestep: float) -> None:
        self.time = time
        self.timestep = timestep
        self.mosquito_timestep = mosquito_timestep
    
    @abstractmethod
    def run(self) -> Any:
        pass
    
    @abstractmethod
    def tick(self) -> Any:
        pass


class BaselineModel(Model):
    """
    The baseline model from Manore et al.

    Attributes
    ---
    k : int
        The number of patches.

    timestep : float
        The gaps between time in the model.

    movement_dist : Callable[() -> float]
        Function to generate a movement rate for an agent.

    sigma_h_arr : List[float]
        List of sigma_h values (the maximum number of mosquito bites an
        individual can sustain per unit time) for each patch.

    sigma_v_arr : List[float]
        List of sigma_v values (the number of times one mosquito would want to
        bite a host per unit time) for each patch.

    K_v_arr : List[float]
        List of carrying capacity values per patch.

    patch_densities : List[float]
        List of densities of patches (for locations).

    phi_v_arr : List[float]
        List of phi_v values (per capita emergence rates of mosquitoes) per
        patch.

    beta_hv_arr : List[float]
        List of beta_hv values (probability of transmission when a mosquito
        bites a host) per patch.

    beta_vh_arr : List[float]
        List of beta_vh values (probability of transmission when a host is
        bitten by a mosquito) per patch.

    nu_v_arr : List[float]
        List of nu_v values (rate of mosquitoes becoming infectious) per patch.

    mu_v_arr : List[float]
        List of mu_v values (death rate of mosquitoes) per patch.

    r_v_arr : List[float]
        List of r_v values (intrinsic growth rate) per patch.

    num_locations : int
        The number of locations/nodes in the model.

    edge_prob : float
        The probability of connecting two nodes when creating the Erdos-Renyi
        graph.

    num_agents : int
        The number of agents in the model.

    initial_infect_proportion : float
        The number of agents initially infected as a proportion.

    nu_h_dist : Callable[() -> float]
        Function to generate a nu_h value (rate of a host becoming infectious
        to the disease).

    mu_h_dist : Callable[() -> float]
        Function to generate a mu_h value (rate of a host recovering).

    total_time : float
        The total time to run the model for (days).

    mosquito_timestep : float
        The time step to solve the mosquito model forward in time.
    """
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
            "patch_sei": {0: np.zeros((int(total_time/timestep), 3)),
                          1: np.zeros((int(total_time/timestep), 3)),
                          2: np.zeros((int(total_time/timestep), 3))},
            "node_seir": {i: [] for i in range(num_locations)},
        }

        # Initialise network — Erdos-Renyi with n, p
        self.graph = nx.erdos_renyi_graph(num_locations, edge_prob)

        # Initialise nodes — distributed according to patch density
        node_patch_ids = np.random.choice(k,
                                          num_locations,
                                          p=patch_densities)
        activity = Activity(activity_id=0, alpha=1)
        
        for node_id in range(num_locations):
            node = Node(node_id=node_id, activity=activity)
            self.nodes[node_id] = node
            self.graph.nodes[node_id]["node"] = node

        # Initialise patches
        for patch_id in range(k):
            patch = Patch(
                k=patch_id,
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
                                   node=np.random.choice(num_locations),
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

        self.num_infected += np.array(
            [agent.state==DiseaseState.INFECTED for agent in self.agents]
            ).sum()
        self.statistics["total_infected"] += self.num_infected


    def tick(self) -> List[Any]:
        """
        Progress the model forward in time by the time step.

        Returns
        ---
        list
            List of tick-specific statistics to log."""
        # (1) Update disease status of vectors and then hosts
        for patch in self.patches:
            patch.tick()
        
        # (2) Move agents randomly
        for agent in self.agents:
            agent.move()
            
        return []


    def run(self, with_progress=False) -> Tuple[Dict[str, Any], List[int]]:
        """
        Run the model until a certain number of time steps.

        Parameters
        ---
        with_progress : bool
            Whether or not to display tqdm progress during run.

        Returns
        ---
        (dict, list)
            Dictionary of model statistics; list of number of infected agents
            over time.
        """
        ticks = int(self.total_time/self.timestep)
        
        if with_progress:
            for _ in tqdm(range(ticks)):
                self.tick()
                self.time += self.timestep
        else:
            for _ in range(ticks):
                self.tick()
                self.time += self.timestep
        
        return self.statistics, [self.num_infected]


class MovementModel(ABC):
    """Abstract class representing the logic to make an agent move on a network."""

    @abstractmethod
    def move_agent(self) -> Any:
        pass


class BaselineMovementModel(MovementModel):
    """
    A baseline movement model as described in Manore et al.

    Attributes
    ---
    model : Model
        The overall model to use.
    """
    def __init__(self, model: Model) -> None:
        self.model = model
        
        
    def move_agent(self, agent: Agent, rho: float) -> None:
        """
        Move an agent with probability 1 - e^{- delta t * rho}.

        Parameters
        ---
        agent : Agent
            The agent to move.

        rho : float
            The movement rate for the agent.
        """
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


