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
            # self._response_efficacy(), # benefits
            self.barriers,#self._barriers()
            self._self_efficacy(),
            self._cues_to_action()
        ])

        product = np.prod(self.ORs**indicators)
        return product/(1+product)

        
    def _susceptibility(self, s_t: float = None) -> int:
        return 1 if s_t >= self.s_star else 0


    def _cues_to_action(self) -> int:
        connections = self.model.agent_network[self.agent.agent_id]

        if len(connections) == 0:
            prop_itn = 0.5
        else:
            prop_itn = np.sum([self.model.agents[a_id].used_itn_last_night for a_id in connections])/len(connections)

        self.model._omega[self.agent.agent_id] = prop_itn
        return 1 if prop_itn >= self.omega else 0

        
    def _barriers(self) -> int:
        return 1 if self.model.temp >= self.t_crit else 0


    # whether agent is 'used to' using the ITNs
    # HBM formulation: indicator version
    def _self_efficacy(self):
        # if self.model.time < self.model.max_itn_score:
            # s_eff = 0.5
        # else:
        s_eff = min(self.agent.itn_score/self.model.max_itn_score, 1)

        if s_eff >= 0.5:
            s_eff = 1
        else:
            s_eff = 0

        self.model._s_eff[self.agent.agent_id] = s_eff
        return s_eff
        # self.model._s_eff[self.agent.agent_id] = min(self.agent.itn_score/(self.model.max_itn_score/2), 1)
        # return 1 if self.agent.itn_score >= self.model.max_itn_score/2 else 0

    # def _response_efficacy(self):
    #     # otherwise, 1 - proportion of infected + use ITNs in network /
    #     # proportion of infected in network
    #     connections = self.model.agent_network[self.agent.agent_id]
    #
    #     # find num infected
    #     infs = [a_id for a_id in connections if self.model.agents[a_id].state == DiseaseState.INFECTED]
    #     if (num_infs := len(infs)) == 0:
    #         return 0.5
    #
    #     # find consistent ITN users and infected
    #     num_itn = np.sum([self.model.agents[a_id].itn_score == self.model.max_itn_score for a_id in infs])
    #
    #     self.model._chi[self.agent.agent_id] = 1-(num_itn/num_infs)
    #     return 1 if self.model._chi[self.agent.agent_id] > .5 else 0
