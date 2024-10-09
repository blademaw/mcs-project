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
        extrinsic = self._extrinsic()
        susceptibility = self._susceptibility(s_t=s_t) 

        response_eff = self._response_efficacy()
        self_eff = self._self_efficacy()

        # adding stochasticity with a beta distribution
        betas = np.random.beta(a=1, b=10, size=2)

        severity     = gen_beta(self.severity, betas[0])
        costs        = gen_beta(self.costs, betas[1])

        # intrinsic    = gen_beta(self.intrinsic, betas[0])
        # response_eff = gen_beta(self.response_eff, betas[2])
        # response_eff = self._response_efficacy()
        # self_eff     = self._self_efficacy()
        # self_eff     = gen_beta(self.self_eff, betas[3])
        # severity = self.severity
        # costs = self.costs

        # self._s_eff = self_eff

        threat = extrinsic - .5*(severity + susceptibility)
        coping = .5*(response_eff + self_eff) - costs
        # threat = .5*((intrinsic + extrinsic) - (severity + susceptibility))
        # coping = response_eff - costs

        self.model._alpha_t[self.agent.agent_id] = threat
        self.model._alpha_c[self.agent.agent_id] = coping

        assert -1 <= coping <= 1, f"Coping {coping} is not in [-1,1]"
        assert -1 <= threat <= 1, f"Threat {threat} is not in [-1,1]"

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

    
    def _self_efficacy(self):
        # whether agent is 'used to' using the ITNs
        s_eff = min(self.agent.itn_score/self.model.max_itn_score, 1)
        self.model._s_eff[self.agent.agent_id] = s_eff
        return s_eff


    def _extrinsic(self):
        connections = self.model.agent_network[self.agent.agent_id]
        prop_itn = (1+np.sum([self.model.agents[a_id].used_itn_last_night for a_id in connections]))/(1+len(connections))
        self.model._omega[self.agent.agent_id] = prop_itn
        if prop_itn >= self.omega: self.model.statistics["count_omega"][self.model.tick_counter] += 1

        return min(prop_itn/self.omega, 1)
        # return 1 if prop_itn >= self.omega else 0


    def _response_efficacy(self):
        # 1 - proportion of infected + use ITNs in network / proportion of infected in network
        connections = self.model.agent_network[self.agent.agent_id]

        # find num infected
        infs = [a_id for a_id in connections if self.model.agents[a_id].state == DiseaseState.INFECTED]
        if (num_infs := len(infs)) == 0:
            return 0.5

        # find ITN users and infected
        num_itn = np.sum([self.model.agents[a_id].itn_score >= self.model.max_itn_score/2 for a_id in infs])

        # if (1-(num_itn/num_infs)) < 0.1: print(num_itn, "/", num_infs)

        self.model._chi[self.agent.agent_id] = 1-(num_itn/num_infs)
        return 1 - (num_itn/num_infs)

