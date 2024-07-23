from enum import Enum
from typing import Set

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

    mode : Model
        The model the agent belongs to.
    """
    def __init__(self,
                 state: DiseaseState,
                 node: int,
                 movement_rate: float,
                 movement_model: 'BaselineMovementModel',
                 nu_h: float,
                 mu_h: float,
                 model: 'Model') -> None:
        self.state = state
        self.node  = node
        self.model = model

        self.nu_h = nu_h
        self.mu_h = mu_h
        
        self.movement_rate  = movement_rate
        self.movement_model = movement_model

        self.num_ticks_in_state = 0


    def move(self) -> None:
        """
        Moves the agent randomly."""
        self.movement_model.move_agent(self, self.movement_rate)


    def update_state(self, lambda_hj: float) -> None:
        """
        Updates the SEIR state of the agent stochastically.

        Parameters
        ---
        lambda_hj : float
            The force of infection on agents for the specific node.
        """
        self.num_ticks_in_state += 1

        r = np.random.random()

        match self.state:
            case DiseaseState.SUSCEPTIBLE:
                self.model.statistics["agent_disease_counts"][0][self.model.tick_counter] += 1

                efficacy = 0
                if self.model.preventive_measures_enabled:
                    if np.random.random() < self.model.adopt_prob:
                        efficacy = .5

                if r < (1 - np.exp(- self.model.timestep * lambda_hj))*(1-efficacy):
                    self.state = DiseaseState.EXPOSED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_exposed"] += 1

            case DiseaseState.EXPOSED:
                self.model.statistics["agent_disease_counts"][1][self.model.tick_counter] += 1
                self.model.statistics["total_time_in_state"][1] += 1
                if r < 1 - np.exp(- self.model.timestep * self.nu_h):
                    self.state = DiseaseState.INFECTED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_infected"] += 1
                    # NOTE: tracking
                    self.model.num_infected += 1

            case DiseaseState.INFECTED:
                self.model.statistics["agent_disease_counts"][2][self.model.tick_counter] += 1
                self.model.statistics["total_time_in_state"][2] += 1
                if r < 1 - np.exp(- self.model.timestep * self.mu_h):
                    self.state = DiseaseState.RECOVERED
                    self.num_ticks_in_state = 0
                    self.model.statistics["total_recovered"] += 1

            case DiseaseState.RECOVERED:
                self.model.statistics["agent_disease_counts"][3][self.model.tick_counter] += 1

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

    agents : Set[Agent]
        The set of all agents in this node.
    """
    def __init__(self,
                 node_id: int,
                 activity: Activity,
                 agents: Set[Agent] | None = None) -> None:
        self.node_id = node_id
        self.activity= activity
        
        self.agents: Set[Agent]  = set() if agents is None else agents
        
        
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
        self.agents.add(agent)
    
    
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the node."""
        self.agents.remove(agent)
