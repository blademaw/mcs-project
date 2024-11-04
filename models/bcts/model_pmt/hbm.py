import numpy as np

from .utils import DiseaseState


def odds_ratio_to_prob(odds_r):
    return odds_r/(1+odds_r)


def gen_beta(fixed_indicator, beta_val):
    if fixed_indicator == 1:
        return 1 - beta_val
    elif fixed_indicator == 0:
        return beta_val
    else:
        raise Exception(f"Indicator is not 0 or 1: {fixed_indicator}")


class ProtectionMotivationTheory:
    """
    Class representing the Protection Motivation Theory (PMT) for an agent.
    """
    def __init__(self,
                 agent: 'Agent',
                 model: 'Model',
                 delta: float,
                 s_star: float,
                 omega: float,
                 severity: int,
                 response_eff: int,
                 self_eff: int,
                 costs: int) -> None:
        self.agent = agent
        self.model = model

        self.delta = delta
        """Time-discounting factor for new cases."""

        self.s_star= s_star
        """Critical $s_t$ to decide whether agent judges 'high' susceptibility."""

        self.omega = omega
        """Minimum proportion of an agent's network that used ITNs last night for 'high' benefits."""

        self.severity = severity
        """Whether maladaptive perceived severity is high for the agent."""

        self.costs = costs
        """Whether adaptive costs are high for the agent."""

        self.response_eff = response_eff
        """Whether adaptive response efficacy is high for the agent."""

        self.self_eff = self_eff

        self.intrinsic = 1
        """Maladaptive intrinsic rewards are always high for agents."""
        
    def compute_prob_behaviour(self, s_t: float) -> float:
        intrinsic = 0.5
        extrinsic = self._extrinsic()
        susceptibility = self._susceptibility(s_t=s_t) 

        response_eff = self._response_efficacy()
        # response_eff = self.response_eff
        self_eff = self._self_efficacy()

        # NOTE: this is the non-stochastic case
        severity = self.severity
        costs    = self.costs

        threat = .5*((intrinsic + extrinsic) - (severity + susceptibility))
        # threat = extrinsic - .5*(severity + susceptibility)
        coping = .5*(self_eff + response_eff) - costs

        self.model._alpha_t[self.agent.agent_id] = threat
        self.model._alpha_c[self.agent.agent_id] = coping

        assert -1 <= coping <= 1, f"Coping {coping} is not in [-1,1]"
        assert -1 <= threat <= 1, f"Threat {threat} is not in [-1,1]"
        return 1/(1+np.e**(-3*(coping-threat)))


        # NOTE: this is probabilistic triggering
        # p_adaptive = 1/(1+np.e**(-3*(coping-threat)))
        # r = np.random.random()
        # if r < p_adaptive:
        #     return odds_ratio_to_prob(1.8 * 2.78)
        # else:
        #     return odds_ratio_to_prob((1/2.69) * .53)

        # NOTE: this is deterministic triggering
        if threat > coping: # maladaptive behaviour
            return odds_ratio_to_prob((1/2.69) * .53) # ORs for high barriers, low benefits
        else: # adaptive behaviour
            # print(f".5*(({self.intrinsic} + {extrinsic})-({self.severity} + {susceptibility}))")
            # print(f".5*({self.response_eff} + {self.self_eff}) - {self.costs}")
            return odds_ratio_to_prob(1.8 * 2.78) # ORs for high severity, high susceptibility

        # threat = 0 if threat <= 0 else 1
        # coping = 0 if coping <= 0 else 1
        #
        # if threat == 0 and coping == 0:
        #     return 0.0
        # elif threat == 0 and coping == 1:
        #     return 1.8/2.8
        # elif threat == 1 and coping == 1:
        #     return 1.0
        # else:
        #     return (1.8*2.78)/(1+1.8*2.78)
        # indicators = np.array([
        #     1,
        #     self._susceptibility(s_t=s_t),
        #     self.severity,#self._severity(),
        #     self.benefits,#self._benefits(),
        #     self.barriers,#self._barriers()
        #     self._cues_to_action()
        # ])

        # product = np.prod(self.ORs**indicators)
        # return product/(1+product)
        

        
    def _susceptibility(self, s_t: float = None):
        # return 1 if s_t >= self.s_star else 0
        return min(s_t/self.s_star, 1)

    
    # whether agent is 'used to' using the ITNs
    def _self_efficacy(self):
        # if there is no reliable data yet, set to default
        # if self.model.time < self.model.max_itn_score:
            # s_eff = 0.5
        # else:
        s_eff = min(self.agent.itn_score/self.model.max_itn_score, 1)

        self.model._s_eff[self.agent.agent_id] = s_eff
        return s_eff


    def _extrinsic(self):
        """External rewards of *not* protecting oneself."""
        connections = self.model.agent_network[self.agent.agent_id]
        
        if len(connections) == 0:
            omega = 0.5
        else:
            prop_itn = np.sum([self.model.agents[a_id].used_itn_last_night for a_id in connections])/len(connections)
            omega = max(1-prop_itn/self.omega, 0)
            self.model._omega[self.agent.agent_id] = omega

        # if omega >= self.omega: self.model.statistics["count_omega"][self.model.tick_counter] += 1
        return omega
        # return 1 if prop_itn >= self.omega else 0


    def _response_efficacy(self):
        # connections = self.model.agent_network[self.agent.agent_id]
        #
        # itn_users = [a_id for a_id in connections if self.model.agents[a_id].used_itn_last_night]
        # if len(itn_users) == 0:
        #     return 0.5
        #
        # not_infs  = len([a_id for a_id in itn_users if self.model.agents[a_id].state in (DiseaseState.SUSCEPTIBLE, DiseaseState.EXPOSED)])
        # chi = not_infs/len(itn_users)
        #
        # self.model._chi[self.agent.agent_id] = chi
        # return chi

        # susc = [a_id for a_id in connections if self.model.agents[a_id].state in (DiseaseState.SUSCEPTIBLE, DiseaseState.EXPOSED)]
        #
        # if (num_susc := len(susc)) == 0:
        #     return 0.5
        #
        # num_itns = np.sum([self.model.agents[a_id].itn_score >= self.model.max_itn_score/2 for a_id in susc])
        # self.model._chi[self.agent.agent_id] = num_itns/num_susc
        # return num_itns/num_susc

        # if there is no reliable data yet, set to default
        # if self.model.time < self.model.max_itn_score:
            # chi = 0.5
        # else:
        # otherwise, 1 - proportion of infected + use ITNs in network /
        # proportion of infected in network
        connections = self.model.agent_network[self.agent.agent_id]

        # find num infected
        infs = [a_id for a_id in connections if self.model.agents[a_id].state == DiseaseState.INFECTED]
        if (num_infs := len(infs)) == 0:
            return 0.5

        # find consistent ITN users and infected
        # num_itn = np.sum([self.model.agents[a_id].itn_score > 0 for a_id in infs])
        # num_itn = np.sum([self.model.agents[a_id].itn_score >= self.model.max_itn_score/2 for a_id in infs])
        # num_itn = np.sum([self.model.agents[a_id].itn_score == self.model.max_itn_score for a_id in infs])
        num_itn = np.sum([self.model.agents[a_id].used_itn_last_night for a_id in infs])
        chi = 1-(num_itn/num_infs)

        self.model._chi[self.agent.agent_id] = chi
        return chi

