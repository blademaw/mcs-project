from typing import List
import numpy as np
from .utils import DiseaseState

class HealthBeliefModel:
    """
    Class representing the Health Belief Model (HBM) for an agent.
    """
    def __init__(self,
                 agent: 'Agent',
                 model: 'Model',
                 ORs: List[float],
                 delta: float,
                 s_star: float,
                 chi: float,
                 omega: float,
                 t_crit: float,
                 severity: bool,
                 benefits: bool,
                 barriers: bool) -> None:
        self.agent = agent
        self.model = model

        self.ORs = ORs
        """Odds ratios for the HBM implementation."""
        
        self.delta = delta
        """Time-discounting factor for new cases."""
        self.s_star= s_star
        """Critical $s_t$ to decide whether agent judges 'high' susceptibility."""
        
        self.chi = chi
        """Minimum proportion of an agent's immediate network with the disease for 'high' severity."""
        
        self.omega = omega
        """Minimum proportion of an agent's network that used ITNs last night for 'high' benefits."""
        
        self.t_crit = t_crit
        """Minimum temperature for an agent to declare 'high' barrier to ITN use."""

        self.severity = severity
        """Whether perceived severity is high for the agent."""

        self.barriers = barriers
        """Whether perceived barriers are high for the agent."""

        self.benefits = benefits
        """Whether perceived benefits are high for the agent."""
        
    def compute_prob_behaviour(self, s_t: float = None) -> float:
        indicators = np.array([
            1,
            self._susceptibility(s_t=s_t),
            self.severity,#self._severity(),
            self.benefits,#self._benefits(),
            self.barriers,#self._barriers()
            self._cues_to_action()
        ])

        product = np.prod(self.ORs**indicators)
        return product/(1+product)

        
    def _susceptibility(self, s_t: float = None) -> int:
        return 1 if s_t >= self.s_star else 0


    def _cues_to_action(self) -> int:
        connections = self.model.agent_network[self.agent.agent_id]
        prop_itn = (1+np.sum([self.model.agents[a_id].used_itn_last_night for a_id in connections]))/(1+len(connections))
        self.model._omega[self.agent.agent_id] = prop_itn
        return 1 if prop_itn >= self.omega else 0

        
    def _barriers(self) -> int:
        return 1 if self.model.temp >= self.t_crit else 0

