import numpy as np
from scipy import integrate


class MosquitoModel:
    """
    Class representing a system of ODEs to be solved for a patch model.

    Attributes
    ---
    patch_id : int
        ID of the patch for the mosquito model.

    K_v : float
        Carrying capacity of the patch.

    phi_v : float
        Per capita emergence rates of mosquitoes for the patch.

    r_v : float
        Intrinsic growth rate of mosquitoes for the patch.

    mu_v : float
        Death rate of mosquitoes for the patch.

    nu_v : float
        Rate of exposed mosquitoes becoming infectious in the patch.

    time : float
        Current time of the model.

    timestep : float
        Time step of the model.

    solve_timestep : float
        Time step to solve the mosquito model forward in time.

    model : Model
        Overall model used.
    """
    def __init__(self,
                 patch_id: int,
                 K_v: float,
                 phi_v: float,
                 r_v: float,
                 mu_v: float,
                 nu_v: float,
                 time: float,
                 timestep: float,
                 solve_timestep: float,
                 model: 'Model'
                ):
        self.patch_id = patch_id
        self.S, self.E, self.I = K_v, 0, 0

        self.N_v = K_v
        self.K_v = K_v
        self.phi_v = phi_v
        self.r_v = r_v
        self.mu_v = mu_v
        self.nu_v = nu_v

        self.time = time
        self.timestep = timestep
        self.solve_timestep = solve_timestep

        self.lambda_v = None
        self.model = model


    def tick(self, lambda_v) -> None:
        """
        Solve the mosquito model forward in time by the required amount.

        Parameters
        ---
        lambda_v : float
            Force of infection on mosquitoes.
        """
        self.lambda_v = lambda_v

        # Determine time range to solve forward
        t   = np.arange(self.time,
                        self.time+self.timestep+self.solve_timestep,
                        self.solve_timestep)

        # Solve forward
        res = integrate.odeint(self._sei_rates,
                               (self.S, self.E, self.I),
                               t)
        self.N_v = res[-1].sum()
        self.S, self.E, self.I = res[-1].T

        self.model.statistics["patch_sei"][self.patch_id][self.model.tick_counter] = res[-1].T


    def _sei_rates(self, X, t) -> np.ndarray:
        """Passed to scipy.integrate.odeint for integration."""
        S, E, I = X
        N_v     = X.sum()

        h_v = (self.phi_v - self.r_v*N_v/self.K_v)*N_v
        
        dS = h_v - self.lambda_v * S - self.mu_v * S
        dE = self.lambda_v * S - self.nu_v * E - self.mu_v * E
        dI = self.nu_v * E - self.mu_v * I

        return np.array([dS, dE, dI])



