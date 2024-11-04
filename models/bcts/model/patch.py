from typing import Set

import numpy as np

from .agent import Node
from .mosquito_model import MosquitoModel


class Patch:
    """
    Represents an area covering nodes with a corresponding mosquito 'cloud'.

    Attributes
    ---
    k : int
        The ID of the patch.

    K_v : float
        The carrying capacity of the patch.

    sigma_v : float
        The number of times one mosquito would want to bite a host per unit time.

    sigma_h : float
        The maximum number of mosquito bites an individual can sustain per unit time.

    psi_v : float
        Per capita emergence of mosquitoes.

    beta_hv : float
        Probability of transmission when a mosquito bites a host.

    beta_vh : float
        Probability of transmission when a host is bitten by a mosquito.

    nu_v : float
        Rate of mosquitoes becoming infectious.

    mu_v : float
        Death rate of mosquitoes.

    r_v : float
        Intrinsic growth rate.

    model : Model
        The overall model.

    nodes : Set[Node]
        The set of nodes for the patch.

    field_node : int
        Field/plantation node for the patch.
    """
    def __init__(self,
                 k: int,
                 K_v: float,
                 sigma_v: float,
                 sigma_h: float,
                 psi_v: float,
                 beta_hv: float,
                 beta_vh: float,
                 nu_v: float,
                 mu_v: float,
                 r_v: float,
                 model: 'Model',
                 nodes: Set[Node] | None = None,
                 field_node: Node | None = None,
                 init_infect_vector_prop: float = 0) -> None:
        self.k = k
        """ID of the patch."""

        self.K_v = K_v
        """$K_v$, the carrying capacity of the patch."""

        self.orig_sigma_v = sigma_v
        """The regular $\sigma_v$ value, the maximum mosquito bite demand per day."""

        self.sigma_v = sigma_v
        """$\sigma_v$, the (current) maximum mosquito bite demand per day."""

        self.sigma_h = sigma_h
        """$\sigma_h$, the maximum number of bites on a mosquito per day."""

        self.model = model

        self.nodes: Set[Node] = set() if nodes is None else nodes
        self.household_nodes = [node for node in self.nodes if node.activity.activity_id == 0]

        self.field_node = field_node
        self.mosquito_model = MosquitoModel(patch_id=k,
                                            K_v=K_v,
                                            psi_v=psi_v,
                                            r_v=r_v,
                                            mu_v=mu_v,
                                            nu_v=nu_v,
                                            time=model.time,
                                            timestep=model.timestep,
                                            solve_timestep=model.mosquito_timestep,
                                            model=model,
                                            init_infect_prop=init_infect_vector_prop
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
        """$\beta_{hv}$, the probability of mosquito-to-human transmission."""

        self.beta_vh = beta_vh
        """$\beta_{vh}$, the probability of human-to-mosquito transmission."""

        self.b   = None
        """$b$, the total number of successful bites between mosquitoes and agents in the patch."""
        self.b_v = None
        self.b_h = None


    def tick(self, sigma_v_modifier):
        """Advance the patch model by one time step."""
        self.sigma_v = self.orig_sigma_v*sigma_v_modifier

        # Update patch values from ABM (agent statistics)
        self._update_patch_values()

        # Advance the patch model (EBM)
        lambda_v = self.get_force_on_vectors()
        self.model.statistics["lambda_v"][self.k].append(lambda_v)
        self.mosquito_model.tick(lambda_v)

        # For each node in this patch, progress disease states of agents
        lambda_hj=None
        for node in self.nodes:
            lambda_hj = node.get_force_on_hosts(b_h=self.b_h,
                                                beta_hv=self.beta_hv,
                                                I_v=self.mosquito_model.I,
                                                N_v=self.mosquito_model.N_v)

            rs = np.random.random(size=len(node.agent_ids))
            for r, agent_id in zip(rs, node.agent_ids):
                self.model.agents[agent_id].update_state(r, lambda_hj)


    def _update_patch_values(self) -> None:
        """Updates the internal values of a patch."""
        seirs, seirs_hat = self._count_agents_in_patch()
        self.N_h, self.N_hat_h = seirs.sum(), seirs_hat.sum()
        
        [self.S_h, self.E_h, self.I_h, self.R_h] = seirs
        [self.S_hat_h, self.E_hat_h, self.I_hat_h, self.R_hat_h] = seirs_hat

        m = self.mosquito_model # for brevity
        
        self.b   = (self.sigma_v * m.N_v * self.sigma_h * self.N_hat_h)/(self.sigma_v * m.N_v + self.sigma_h * self.N_hat_h)
        self.b_v = self.b/m.N_v
        self.b_h = self.b/self.N_hat_h

        self.model.statistics["patch_values"][self.model.tick_counter*(self.model.k) + self.k] = {"patch_id": self.k,
                                                                                                  "b": self.b,
                                                                                                  "b_v": self.b_v,
                                                                                                  "b_h": self.b_h}


    def _count_agents_in_patch(self):
        """Counts agents in this patch."""
        seirs     = np.array([0., 0., 0., 0.])
        seirs_hat = np.array([0., 0., 0., 0.])

        for node in self.nodes:
            seirs     += node.seirs
            seirs_hat += node.activity.alpha * node.seirs

        return seirs, seirs_hat


    def get_force_on_vectors(self) -> float:
        """
        Calculate the force of infection on vectors for this patch (lambda_v).

        Returns
        ---
        float
            The force of infection on vectors (lambda_v)."""
        return self.b_v * self.beta_vh * (self.I_hat_h/self.N_hat_h)
