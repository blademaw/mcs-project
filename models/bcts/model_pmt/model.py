from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from .agent import Activity, Agent, Node
from .hbm import ProtectionMotivationTheory
from .patch import Patch
from .utils import DiseaseState

# Define activities used for location types
HOUSEHOLD_ACTIVITY   = Activity(activity_id=0, alpha=.43)
FOREST_SITE_ACTIVITY = Activity(activity_id=1, alpha=1.)
FIELD_SITE_ACTIVITY  = Activity(activity_id=2, alpha=1.)

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

    psi_v_arr : List[float]
        List of psi_v values (per capita emergence rates of mosquitoes) per
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

    num_households: int
        The number of households in the model.

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
                 movement_dist: Callable[..., float],
                 sigma_h_arr: np.ndarray,
                 sigma_v_arr: np.ndarray,
                 K_v_arr: np.ndarray,
                 patch_densities: np.ndarray,
                 psi_v_arr: np.ndarray,
                 beta_hv_arr: np.ndarray,
                 beta_vh_arr: np.ndarray,
                 nu_v_arr: np.ndarray,
                 mu_v_arr: np.ndarray,
                 num_households: int,
                 edge_prob: float,
                 num_agents: int,
                 initial_infect_proportion: float,
                 patch_init_infect_vector_prop_arr: List[float],
                 nu_h_dist: Callable[..., float],
                 mu_h_dist: Callable[..., float],
                 total_time: float,
                 mosquito_timestep: float,
                 high_severity: bool,
                 high_response_eff: bool,
                 high_self_eff: bool,
                 high_response_costs: bool,
                 prob_adopt_itn: float = .0,
                 forest_worker_prob: float = .05,
                 field_worker_prob: float = .71,
                 stay_home_chance: float = 0.0,
                 init_itn_score: float = 0.0) -> None:
        # Assign model-specific parameters
        self.time = 0.0
        """The current time of the model, in days."""

        self.max_itn_score = 10

        self.k = k
        self.tick_counter = 0
        """The current tick."""

        self.timestep = timestep
        """The timestep amount (in days) of the model."""

        self.total_time = total_time
        """The maximum number of time (in days) the model runs for."""

        self.mosquito_timestep = mosquito_timestep
        """The timestep of the mosquito model."""

        self.movement_model = BaselineMovementModel(model=self)
        self.num_agents = num_agents
        self.num_households = num_households
        self.num_locations = num_households + 1 + 2 # 1 forest site + 2 fields
        """The total number of locations in the model."""

        self.K_v_arr = K_v_arr
        """The array of `K_v`, mosquito carrying capacities per patch."""

        # Parameters for mobility and demographics
        assert (forest_worker_prob < 1 and field_worker_prob < 1) and (forest_worker_prob + field_worker_prob <= 1), "Forest and field worker probabilities must sum to at most 1."
        self.stay_home_chance = stay_home_chance
        """The chance for non-working agents to remain in their households during the day."""

        self._base_temp = None
        self.temp = None
        """The current temperature in the model."""

        self.c_t = [np.zeros(int(self.total_time/self.timestep)) for _ in range(k)]
        """The number of new illnesses at timestep i."""
        self._chi = np.zeros(num_agents)
        self._omega = np.zeros(num_agents)
        self._s_eff = np.zeros(num_agents)
        self._alpha_t = np.zeros(num_agents)
        self._alpha_c = np.zeros(num_agents)

        self.forest_worker_prob = forest_worker_prob
        self.field_worker_prob = field_worker_prob
        self.prob_adopt_itn = prob_adopt_itn
        self.asleep = False
        """Whether agents should be asleep currently in the model."""

        # Entities
        self.agents:  List[Agent] = np.full(num_agents,
                                            fill_value=None,
                                            dtype=Agent)
        self.nodes:   List[Node]  = np.full(self.num_locations,
                                            fill_value=None,
                                            dtype=Node)
        self.patches: List[Patch] = np.full(k, fill_value=None, dtype=Patch)
        
        self.num_infected = 0
        self.statistics = {
            # general statistics needed for plots:
            "time": [],
            "lambda_v": [[], [], []],
            "agent_disease_counts": np.zeros((3,
                                              4,
                                              int(self.total_time/self.timestep))),
            "agent_infected_unique": np.zeros((3, int(self.total_time/self.timestep))),
            "patch_values": [None for _ in range(k*int(self.total_time/self.timestep))],
            "infection_records": [],
            "total_infected": 0,
            "patch_sei": {0: np.zeros((int(total_time/timestep), 3)),
                1: np.zeros((int(total_time/timestep), 3)),
                2: np.zeros((int(total_time/timestep), 3))},

            # PMT-specific
            "prob_values": np.zeros(int(self.total_time/self.timestep)),
            "chi": np.zeros(int(self.total_time/self.timestep)),
            "omega": np.zeros(int(self.total_time/self.timestep)),
            "self_eff": np.zeros(int(self.total_time/self.timestep)),
            "alpha_t": np.zeros(int(self.total_time/self.timestep)),
            "alpha_c": np.zeros(int(self.total_time/self.timestep)),

            # NOTE: unused stats; don't collect
            # "lambda_hj": [],
            # "num_infected": {
            #     0: [],
            #     1: [],
            #     2: []
            # },
            # 0 = forest worker;  1 = field worker; 2 = non-worker
            # "time_in_household": 0,
            # "time_in_field": 0,
            # "temperature": np.zeros(int(self.total_time/self.timestep)),
            # "used_itn": np.zeros(int(self.total_time/self.timestep)),
            # "num_movements": 0,
            # "total_exposed": 0,
            # "total_recovered": 0,
            # "total_time_in_state": [0, 0, 0, 0],
            # "node_seir": {i: [] for i in range(self.num_locations)},
            # "count_omega": np.zeros(int(self.total_time/self.timestep)), # number of people who have >= omega contacts using ITNs
        }


        # Initialise network — Erdos-Renyi with n, p
        self.graph = nx.erdos_renyi_graph(self.num_locations, edge_prob)

        # Initialise households — distributed according to patch density
        household_patch_ids = np.random.choice(k,
                                          num_households,
                                          p=patch_densities)
        cur_node_id = num_households

        # Add households to model
        for node_id in range(num_households):
            self._add_node(node_id, household_patch_ids[node_id], HOUSEHOLD_ACTIVITY)

        # Initialise patches
        for i in range(k):
            patch = Patch(
                k=i,
                K_v=K_v_arr[i],
                sigma_v=sigma_v_arr[i],
                sigma_h=sigma_h_arr[i],
                psi_v=psi_v_arr[i],
                beta_hv=beta_hv_arr[i],
                beta_vh=beta_vh_arr[i],
                nu_v=nu_v_arr[i],
                mu_v=mu_v_arr[i],
                r_v=psi_v_arr[i]-mu_v_arr[i],
                model=self,
                nodes=set(self.nodes[np.where(household_patch_ids == i)]),
                init_infect_vector_prop=patch_init_infect_vector_prop_arr[i]
            )
            self.patches[i] = patch


        ## Create and add special location nodes
        # 1. Forest work site
        self._add_node(cur_node_id,
                       2, # exists only in patch 2 (inner forest)
                       FOREST_SITE_ACTIVITY)
        self.forest_node = self.nodes[cur_node_id]
        self.patches[2].nodes.add(self.nodes[cur_node_id])
        cur_node_id += 1

        # 2. Plantation/field nodes (patches 0, 1; outer/fringe forest)
        for i in range(2):
            self._add_node(cur_node_id,
                           i,
                           FIELD_SITE_ACTIVITY)
            self.patches[i].field_node = self.nodes[cur_node_id]
            self.patches[i].nodes.add(self.nodes[cur_node_id])
            cur_node_id += 1
        self.patches[2].field_node = self.nodes[cur_node_id-1] # field workers in forest commute to fringe forest


        # Initialise agents
        agent_disease_states = np.random.choice([DiseaseState.SUSCEPTIBLE,
                                                 DiseaseState.INFECTED],
                                                p=[1-initial_infect_proportion,initial_infect_proportion],
                                                size=self.num_agents)
        agent_t_crits = np.random.normal(29, 1, size=self.num_agents)
        
        for i in range(num_agents):
            forest_worker = False
            field_worker = False
            mover = True

            work_node = None
            home_node = self.nodes[np.random.choice(num_households)]

            # Assign worker type
            worker_type = np.random.choice(
                range(3), # forest ; field ; non-worker
                p=[self.forest_worker_prob,
                   self.field_worker_prob,
                   1-self.forest_worker_prob-self.field_worker_prob]
            )

            if worker_type == 0:
                forest_worker = True
                work_node = self.forest_node
            elif worker_type == 1:
                field_worker = True
                work_node = self.patches[self.nodes[home_node.node_id].patch_id].field_node
            else:
                if np.random.random() < self.stay_home_chance:
                    mover = False

            agent = Agent(agent_id=i,
                          state=agent_disease_states[i],
                          node=home_node, # all agents start in their home node
                          movement_rate=movement_dist(),
                          movement_model=self.movement_model,
                          nu_h=nu_h_dist(),
                          mu_h=mu_h_dist(),
                          forest_worker=forest_worker,
                          field_worker=field_worker,
                          home_node=home_node,
                          model=self,
                          work_node=work_node,
                          mover=mover,
                          itn_score=init_itn_score,
                          )
            self.agents[i] = agent
            agent.node.add_agent(agent)

        self.num_infected += np.array(
            [agent.state==DiseaseState.INFECTED for agent in self.agents]
            ).sum()
        self.statistics["total_infected"] += self.num_infected
        self.statistics["agent_group_pops"] = [np.sum([a.node.patch_id == j for a in self.agents]) for j in range(3)]

        # Opitmisation: precompute choices for non-workers to move given node
        self.move_choices = [[] for _ in range(num_households)]
        for node_id in range(num_households):
            self.move_choices[node_id] = [dest_id for dest_id in self.graph.adj[node_id] if dest_id < self.num_households and self.nodes[dest_id].patch_id == self.nodes[node_id].patch_id]

        # Create agent social network
        # self.agent_network = nx.Graph()
        # self.agent_network.add_nodes_from(list(range(self.num_agents)))
        self.agent_network = [[] for _ in range(self.num_agents)]
        for a1 in range(num_agents):
            connected_locs = self.move_choices[self.agents[a1].node.node_id]
            self.agent_network[a1] = [a2 for a2s in [self.nodes[i].agent_ids for i in connected_locs] for a2 in a2s]
            # self.agent_network.add_edges_from([(a1, a2) for a2 in connected_agent_ids])

        # Check that all patches know the nodes they have
        assert sorted([i for l in [[n.node_id for n in p.nodes] for p in self.patches] for i in l]) == list(range(self.num_households + 3)), "At least one node is unaccounted for in a patch."

        # Set HBM
        self.s_stars = np.array([np.sum([a.node.patch_id == j for a in self.agents]) for j in range(3)])*.02

        for agent in self.agents:
            self._hbm_delta_val = .99
            hbm = ProtectionMotivationTheory(agent=agent,
                                    model=self,
                                    delta=self._hbm_delta_val,
                                    s_star=self.s_stars[agent.home_node.patch_id],
                                    omega=.5,
                                    severity=int(high_severity),
                                    response_eff=int(high_response_eff),
                                    self_eff=int(high_self_eff),
                                    costs=int(high_response_costs))
            agent.hbm = hbm


    def _add_node(self, node_id: int,
                  patch_id: int,
                  activity: Activity) -> None:
        node = Node(node_id=node_id, patch_id=patch_id, activity=activity)
        self.nodes[node_id] = node
        self.graph.nodes[node_id]["node"] = node


    def tick(self) -> List[Any]:
        """
        Progress the model forward in time by the time step.

        Returns
        ---
        list
            List of tick-specific statistics to log."""
        # (0) Update temperature
        time_in_hrs = self.tick_counter*(self.timestep*24) % 24
        # if time_in_hrs == 0:
        #     # new day = new random base temperature
        #     self._temp_base = np.random.normal(25, 2)
        # self.temp = self._temp_base + 5*np.sin((np.pi/11)*time_in_hrs - .8*np.pi)
        # self.statistics["temperature"][self.tick_counter] = self.temp

        # HACK: generate s_t now because assume all agents have same delta
        s_t = [0 for _ in range(self.k)]
        for patch_id in range(len(self.patches)):
            s_t[patch_id] = np.sum([(self._hbm_delta_val**i)*(self.c_t[patch_id][self.tick_counter-i-1]) for i in range(self.tick_counter)])

        # (1) Update disease status of vectors and then hosts

        # If dawn/night/dusk, strengthen sigma_v (mosquito aggressiveness by 5x)
        # if (self.time*24 % 24 >= 18) or (self.time*24 % 24 <= 8):
        sigma_v_modifier = 1
        if (time_in_hrs >= 18) or (time_in_hrs <= 8):
            sigma_v_modifier = 4

        for patch in self.patches:
            patch.tick(sigma_v_modifier=sigma_v_modifier)

        # (2) Check if agents should be sleeping, move agents
        # if (self.time*24 % 24 >= 18) or (self.time*24 % 24 <= 6):
        if (time_in_hrs >= 18) or (time_in_hrs <= 4):
            if not self.asleep:
                # Generate random numbers for agents
                p_vals     = np.zeros(self.num_agents)
                adopt_itns = np.random.random(size=self.num_agents)

                # Agents go to their home node and are asleep
                for agent in self.agents:
                    self.graph.nodes[agent.node.node_id]["node"].remove_agent(agent)
                    self.graph.nodes[agent.home_node.node_id]["node"].add_agent(agent)
                    
                    agent.node = agent.home_node

                    # If agents adopt ITNs, set to active
                    p = agent.hbm.compute_prob_behaviour(s_t=s_t[agent.home_node.patch_id]) # HACK: generating s_t globally
                    p_vals[agent.agent_id] = p
                    self.statistics["chi"][self.tick_counter] = np.mean(self._chi)
                    self.statistics["omega"][self.tick_counter] = np.mean(self._omega)
                    self.statistics["self_eff"][self.tick_counter] = np.mean(self._s_eff)
                    self.statistics["alpha_t"][self.tick_counter] = np.mean(self._alpha_t)
                    self.statistics["alpha_c"][self.tick_counter] = np.mean(self._alpha_c)
                    
                    # agents use ITNs
                    if adopt_itns[agent.agent_id] < p:
                        agent.itn_score = min(agent.itn_score+1, self.max_itn_score)
                        agent.itn_active = True
                        agent.used_itn_last_night = True

                        # self.statistics["used_itn"][self.tick_counter] += 1
                    # agents don't use ITNs
                    else:
                        agent.itn_score = max(agent.itn_score-.25, 0)
                        agent.used_itn_last_night = False
                
                self.statistics["prob_values"][self.tick_counter] = np.mean(p_vals)

                self.asleep = True
        else:
            # Agents are awake
            self.asleep = False

            for agent in self.agents:
                # When awake, ITN protection is off
                agent.itn_active = False
                agent.move()

        self.tick_counter += 1
        self.statistics["time"] += [self.time]
        return []


    def run(self,
            with_progress=False) -> Tuple[Dict[str, Any], List[int]]:
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

    Methods
    ---
    move_agent
        Move an agent in the model.

    move_work
        Move a worker agent according to the worker agent movement model.

    move_random
        Move an agent according to the random movement model (from Manore et
        al. (2015)).
    """
    def __init__(self, model: Model) -> None:
        self.model = model


    def move_agent(self, agent: Agent, rho: float) -> None:
        """
        Move an agent in the simulation. Decides how agents move in the
        model. Agents sleep between the hours of 6pm--6am, and move outside
        these hours.

        Parameters
        ---
        rho : float
            The movement rate of the agent.
        """
        if agent.forest_worker or agent.field_worker:
            self.move_work(agent)
        else:
            self.move_random(agent, rho)


    def move_work(self, agent: Agent) -> None:
        """
        Move a worker agent to their place of work, or home.

        Parameters
        ---
        agent : Agent
            The worker agent to be moved.
        """
        assert agent.work_node is not None, f"`move_work()` triggered on non-working agent with node {agent.work_node.node_id} and worker={agent.forest_worker}"

        if agent.node.node_id != agent.work_node.node_id:
            # Worker agents go to work
            self.model.graph.nodes[agent.node.node_id]["node"].remove_agent(agent)
            self.model.graph.nodes[agent.work_node.node_id]["node"].add_agent(agent)
            
            agent.node = agent.work_node
            

    def move_random(self, agent: Agent, rho: float) -> None:
        """
        Move an agent with probability 1 - e^{- delta t * rho}.

        Parameters
        ---
        agent : Agent
            The agent to move.

        rho : float
            The movement rate for the agent.
        """
        if agent.mover and np.random.random() < (1 - np.exp(-self.model.timestep*rho)):
            # If an agent decides to move
            # self.model.statistics["num_movements"] += 1

            # Move to a household uniformly
            choices = self.model.move_choices[agent.node.node_id]

            # Only if an agent can actually move
            if len(choices) > 0:
                new_node = self.model.nodes[np.random.choice(choices)]

                self.model.graph.nodes[agent.node.node_id]["node"].remove_agent(agent)
                self.model.graph.nodes[new_node.node_id]["node"].add_agent(agent)
                agent.node = new_node
