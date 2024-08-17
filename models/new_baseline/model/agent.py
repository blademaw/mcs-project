from enum import Enum
from typing import Set, Union

import numpy as np

class DiseaseState(Enum):
    """An enum for different disease states."""
    SUSCEPTIBLE = 0
    EXPOSED     = 1
    INFECTED    = 2
    RECOVERED   = 3

class Agent:
    """A class representing an agent on a network.

    Attributes
    ---
    state : DiseaseState
        The disease state of the agent.

    node : int
        The ID of the node the agent is in.

    movement_rate : float
        The movement rate to pass to the movement model.

    movement_model : MovementModel
        The underlying movement model that moves the agent.

    nu_h : float
        The rate of becoming infectious.

    mu_h : float
        The rate of recovery.

    forest_worker : bool
        Whether the agent is a forest worker (commutes to the forest each day
        to work).

    field_worker : bool
        Whether the agent is a field worker (commits to their home patch field
        during the day to work).

    home_node : int
        The node the agent resides in.

    model : Model
        The model the agent belongs to.
    """
    def __init__(self,
                 agent_id: int,
                 state: DiseaseState,
                 node: 'Node',
                 movement_rate: float,
                 movement_model: 'BaselineMovementModel',
                 nu_h: float,
                 mu_h: float,
                 forest_worker: bool,
                 field_worker: bool,
                 home_node: 'Node',
                 model: 'Model',
                 work_node: Union['Node', None] = None,
                 mover: bool = True) -> None:
        self.agent_id = agent_id
        self.state = state
        self.node  = node
        self.model = model
        self.home_node = home_node
        self.work_node = work_node

        self.nu_h = nu_h
        """$\nu_h$, the rate at which agents become infectious."""

        self.mu_h = mu_h
        """$\mu_h$, the rate at which agents recover."""
        
        self.movement_rate  = movement_rate
        """$\rho$, the rate at which agents move during a day."""

        self.movement_model = movement_model
        self.mover = mover
        """Whether the (non-working) agent moves or stays at home during the day."""

        self.forest_worker = forest_worker
        self.field_worker = field_worker
        self.itn_active = False

        self.num_ticks_in_state = 0

        # this saves me from deriving it each timestep
        # TODO: convert this to an enum.
        self._worker_type = 0 if forest_worker else 1 if field_worker else 2
        """Type of worker. 0 is a forest worker; 1 is field worker; 2 is non-worker."""


    def move(self) -> None:
        """
        Moves the agent randomly.
        """
        self.movement_model.move_agent(self, self.movement_rate)


    def update_state(self, r: float, lambda_hj: float) -> None:
        """
        Updates the SEIR state of the agent stochastically.

        Parameters
        ---
        r : float
            The random number generated for the agent (0 < r < 1)

        lambda_hj : float
            The force of infection on agents for the specific node.
        """
        self.num_ticks_in_state += 1

        match self.state:
            case DiseaseState.SUSCEPTIBLE:
                self.model.statistics["agent_disease_counts"][self._worker_type][0][self.model.tick_counter] += 1

                efficacy = .99 if self.itn_active else 0
                if r < (1 - np.exp(- self.model.timestep * lambda_hj))*(1-efficacy):
                    self.node.seirs[0] -= 1
                    self.node.seirs[1] += 1
                    
                    self.state = DiseaseState.EXPOSED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_exposed"] += 1
                    self.model.statistics["infection_records"] += [{"time": self.model.tick_counter,
                                                                    "patch": self.node.patch_id,
                                                                    "worker_type": self._worker_type,
                                                                    "activity_id": self.node.activity.activity_id,
                                                                    "home_patch": self.home_node.patch_id}]

            case DiseaseState.EXPOSED:
                self.model.statistics["agent_disease_counts"][self._worker_type][1][self.model.tick_counter] += 1
                self.model.statistics["total_time_in_state"][1] += 1

                # Mosquito biting agent subprocess
                if r < 1 - np.exp(- self.model.timestep * self.nu_h):
                    self.node.seirs[1] -= 1
                    self.node.seirs[2] += 1

                    self.state = DiseaseState.INFECTED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_infected"] += 1
                    self.model.statistics["agent_infected_unique"][self._worker_type][self.model.tick_counter] += 1
                    # self.model.statistics["infection_records"] += [{"time": self.model.tick_counter,
                    #                                                 "patch": self.node.patch_id,
                    #                                                 "worker_type": self._worker_type,
                    #                                                 "activity_id": self.node.activity.activity_id}]
                    # NOTE: tracking
                    self.model.num_infected += 1

            case DiseaseState.INFECTED:
                self.model.statistics["agent_disease_counts"][self._worker_type][2][self.model.tick_counter] += 1
                self.model.statistics["total_time_in_state"][2] += 1
                if r < 1 - np.exp(- self.model.timestep * self.mu_h):
                    self.node.seirs[2] -= 1
                    self.node.seirs[3] += 1

                    self.model.c_t[self.model.tick_counter] += 1

                    self.state = DiseaseState.RECOVERED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_recovered"] += 1

            case DiseaseState.RECOVERED:
                self.model.statistics["agent_disease_counts"][self._worker_type][3][self.model.tick_counter] += 1

            case _:
                pass


class Activity:
    """
    Class to define an activity.

    Attributes
    ---
    activity_id : int
        The ID of the activity.

    alpha : float
        The exposure amount of the activity. In [0,1].
    """
    def __init__(self, activity_id: int, alpha: float) -> None:
        self.activity_id = activity_id
        
        assert alpha >= 0 and alpha <= 1, "Alpha must be in [0,1]"
        self.alpha = alpha


class Node:
    """
    A class representing a location (node) in the network model.

    Attributes
    ---
    node_id : int
        The ID of the node.

    activity : Activity
        The activity belonging to the node.

    patch_id : int
        The ID of the node's patch.

    agents : Set[int]
        The set of all agents (IDs) in this node.
    """
    def __init__(self,
                 node_id: int,
                 patch_id: int,
                 activity: Activity,
                 agent_ids: Set[int] = None) -> None:
        self.node_id = node_id
        self.patch_id = patch_id
        self.activity = activity

        self.agent_ids: Set[int] = set() if agent_ids is None else agent_ids

        self.seirs = np.zeros(4)
        """SEIR values for a node (no need for counting each tick)."""
        
        
    def get_force_on_hosts(self,
                           b_h: float,
                           beta_hv: float,
                           I_v: float,
                           N_v: float) -> float:
        """
        Calculate the force of infection on hosts/agents for this node
        (lambda_{h,j}).

        Parameters
        ---
        b_h : float
            The biting rate on hosts.

        beta_hv : float
            The probability of infection from an infectious mosquito to a
            susceptible host.

        I_v : float
            The number of infectious mosquitoes.

        N_v : float
            The total number of infectious mosquitoes.

        Returns
        ---
        float
            The force of infection on hosts, lambda_{h,j}.
        """
        return (self.activity.alpha * b_h) * beta_hv * (I_v/N_v)
    
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the node."""
        self.seirs[agent.state.value] += 1
        self.agent_ids.add(agent.agent_id)


    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the node."""
        self.seirs[agent.state.value] -= 1
        self.agent_ids.remove(agent.agent_id)
