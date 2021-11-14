import numpy as np
from scipy import sparse
import pickle
import itertools
import matplotlib.pyplot as plt


class OneDimBHM_w_RBM:
    def __init__(self, t, mu, N_s, M, n_max, learning_rate, gamma, epscut, P):
        self.t = t  # normalized hopping strength --> t/U
        self.mu = mu  # normalized chemical potential --> 𝜇/U

        self.N_s = N_s  # number of lattice sites
        self.M = M  # number of hidden neurons
        self.n_max = n_max  # maximum number of particles in each lattice site

        self.learning_rate = learning_rate
        self.gamma = gamma  # exponential decay rate
        self.epscut = epscut  # small cutoff value
        self.P = P  # number of samples

        # initializing the network parameters
        self.alpha = None
        self.beta = None
        self.W = None
        self.initialize_network_parameters()

        # determining all states in the Fock space
        self.states_in_FS = np.array(
            list(itertools.product(np.arange(self.n_max + 1), repeat=self.N_s))
        )

        # defining the operators in the hopping part of the Hamiltonian
        self.b_i_dag_b_i_plus_1 = np.identity(self.N_s) - np.diag(
            np.ones(self.N_s - 1, dtype=np.int32), 1
        )
        self.b_i_dag_b_i_plus_1[-1, 0] = -1
        self.b_i_plus_1_dag_b_i = -self.b_i_dag_b_i_plus_1.copy()

        # initializing the states and number of particles array to find connected states in the hopping part
        self.hop_kets = np.empty((0, self.N_s), dtype=np.int32)
        self.hop_bras = np.empty((0, self.N_s), dtype=np.int32)
        self.nums_of_particles_raised_and_lowered_state = np.empty(
            (0, 2), dtype=np.int32
        )
        self.hop_values = np.empty((0, 1), dtype=np.int64)
        self._hop_kets = np.empty((0, self.N_s), dtype=np.int32)
        self._hop_bras = np.empty((0, self.N_s), dtype=np.int32)
        self._nums_of_particles_raised_and_lowered_state = np.empty(
            (0, 2), dtype=np.int32
        )
        self._hop_values = np.empty((0, 1), dtype=np.int64)

        # finding the connected states in the hopping part
        for input_state in self.states_in_FS:
            for state_i in self.states_in_FS:
                tiled_state_i = np.tile(state_i, (self.N_s, 1))

                resulted_states_1 = self.b_i_dag_b_i_plus_1 + tiled_state_i
                resulted_states_2 = self.b_i_plus_1_dag_b_i + tiled_state_i

                for row_index_of_matched_state_1 in np.where(
                    (resulted_states_1 == input_state).all(axis=1)
                )[0]:

                    index_of_the_raised_state_1 = np.where(
                        self.b_i_dag_b_i_plus_1[row_index_of_matched_state_1, :] == 1
                    )[0][0]

                    index_of_the_lowered_state_1 = np.where(
                        self.b_i_dag_b_i_plus_1[row_index_of_matched_state_1, :] == -1
                    )[0][0]

                    num_of_particle_in_the_raised_state_1 = state_i[
                        index_of_the_raised_state_1
                    ]
                    num_of_particle_in_the_lowered_state_1 = state_i[
                        index_of_the_lowered_state_1
                    ]

                    if row_index_of_matched_state_1 == 0:
                        self._hop_kets = np.vstack([self.hop_kets, state_i])
                        self._hop_bras = np.vstack([self.hop_bras, input_state])
                        self._nums_of_particles_raised_and_lowered_state = np.vstack(
                            [
                                self.nums_of_particles_raised_and_lowered_state,
                                [
                                    num_of_particle_in_the_raised_state_1,
                                    num_of_particle_in_the_lowered_state_1,
                                ],
                            ]
                        )

                    self.hop_kets = np.vstack([self.hop_kets, state_i])
                    self.hop_bras = np.vstack([self.hop_bras, input_state])
                    self.nums_of_particles_raised_and_lowered_state = np.vstack(
                        [
                            self.nums_of_particles_raised_and_lowered_state,
                            [
                                num_of_particle_in_the_raised_state_1,
                                num_of_particle_in_the_lowered_state_1,
                            ],
                        ]
                    )
                    self.hop_values = np.append(
                        self.hop_values,
                        np.sqrt(num_of_particle_in_the_raised_state_1 + 1)
                        * np.sqrt(num_of_particle_in_the_lowered_state_1),
                    )

                if self.N_s > 2:
                    for row_index_of_matched_state_2 in np.where(
                        (resulted_states_2 == input_state).all(axis=1)
                    )[0]:

                        index_of_the_raised_state_2 = np.where(
                            self.b_i_plus_1_dag_b_i[row_index_of_matched_state_2, :]
                            == 1
                        )[0][0]

                        index_of_the_lowered_state_2 = np.where(
                            self.b_i_plus_1_dag_b_i[row_index_of_matched_state_2, :]
                            == -1
                        )[0][0]

                        num_of_particle_in_the_raised_state_2 = state_i[
                            index_of_the_raised_state_2
                        ]
                        num_of_particle_in_the_lowered_state_2 = state_i[
                            index_of_the_lowered_state_2
                        ]

                        self.hop_kets = np.vstack([self.hop_kets, state_i])
                        self.hop_bras = np.vstack([self.hop_bras, input_state])
                        self.nums_of_particles_raised_and_lowered_state = np.vstack(
                            [
                                self.nums_of_particles_raised_and_lowered_state,
                                [
                                    num_of_particle_in_the_raised_state_2,
                                    num_of_particle_in_the_lowered_state_2,
                                ],
                            ]
                        )
                        self.hop_values = np.append(
                            self.hop_values,
                            np.sqrt(num_of_particle_in_the_raised_state_2 + 1)
                            * np.sqrt(num_of_particle_in_the_lowered_state_2),
                        )

        # initializing the arrays to store the calculated wavefunctions in each iteration
        self.psi_data_keys = np.empty((0, self.N_s), dtype=np.int32)
        self.psi_data_keys_ = np.empty((0, self.N_s * self.n_max), dtype=np.int32)
        self.psi_data_values = np.empty((0, 1), dtype=np.complex128)

        self.samples = None

    def reset_psi_data(self):
        """Resets the arrays to store the calculated wavefunctions in each iteration."""

        self.psi_data_keys = np.empty((0, self.N_s), dtype=np.int32)
        self.psi_data_keys_ = np.empty((0, self.N_s * self.n_max), dtype=np.int32)
        self.psi_data_values = np.empty((0, 1), dtype=np.complex128)

    def initialize_network_parameters(self):
        """Initializes the network parameters of the RBM randomly."""

        self.alpha = np.random.rand(self.N_s * self.n_max) + 1j * np.random.rand(
            self.N_s * self.n_max
        )
        self.beta = np.random.rand(self.M) + 1j * np.random.rand(self.M)
        self.W = np.random.rand(self.M, self.N_s * self.n_max) + 1j * np.random.rand(
            self.M, self.N_s * self.n_max
        )

    def calc_psi(self, state):
        """Calculates the RBM wavefunction ansatz."""

        try:
            matched_states = np.where((self.psi_data_keys == state).all(axis=1))[0]
        except:
            matched_states = []
        if len(matched_states):
            row_index_of_matched_state = matched_states[0]
            psi = self.psi_data_values[row_index_of_matched_state]
        else:
            one_hot_encoded_state = np.zeros(self.N_s * self.n_max, dtype=np.int32)
            for index, i in enumerate(state):
                one_hot_encoded_state[
                    index * self.n_max : self.n_max * (index + 1)
                ] = np.concatenate(
                    (
                        np.ones(int(i), dtype=np.int32),
                        np.zeros(self.n_max - int(i), dtype=np.int32),
                    ),
                    axis=None,
                )

            psi = np.exp(self.alpha @ one_hot_encoded_state)
            psi = psi * np.prod(
                2 * np.cosh(self.beta + (self.W @ one_hot_encoded_state))
            )

            self.psi_data_keys = np.vstack([self.psi_data_keys, state])
            self.psi_data_keys_ = np.vstack(
                [self.psi_data_keys_, one_hot_encoded_state]
            )
            self.psi_data_values = np.append(self.psi_data_values, psi)

        return psi

    def run_metropolis_hastings_sampler(self):
        """Samples |Ψ|^2 and creates a Markov chain by using Metropolis-Hastings algorithm."""

        # initializing the Markov chain as a matrix of samples
        markov_chain = np.zeros((self.P, self.N_s), dtype=np.int32)
        markov_chain_onehot = np.zeros((self.P, self.N_s * self.n_max), dtype=np.int32)

        # picking the first sample randomly
        first_sample = self.states_in_FS[
            np.random.randint(self.states_in_FS.shape[0]), :
        ]
        markov_chain[0] = first_sample
        for index, i in enumerate(markov_chain[0]):
            markov_chain_onehot[
                0, index * self.n_max : self.n_max * (index + 1)
            ] = np.concatenate(
                (
                    np.ones(int(i), dtype=np.int32),
                    np.zeros(self.n_max - int(i), dtype=np.int32),
                ),
                axis=None,
            )

        # calculating the wavefunction of the first sample in the Markov chain
        psi_of_m_th_state = self.calc_psi(markov_chain[0])

        # drawing samples
        for i in np.arange(1, self.P):
            markov_chain[i] = markov_chain[i - 1].copy()

            # choosing a lattice site randomly
            m = np.random.randint(0, self.N_s)

            # choosing an integer between 0 and n_max randomly
            n_rand = np.random.randint(self.n_max + 1)

            # set the occupation number of the randomly picked lattice site to randomly picked integer
            while markov_chain[i, m] == n_rand:
                n_rand = np.random.randint(self.n_max + 1)

            # defining the candidate state for the Markov chain
            candidate_state = markov_chain[i - 1].copy()
            candidate_state[m] = n_rand

            # drawing a sample from the uniform distribution (0,1)
            R = np.random.uniform()

            # calculating the wavefunction of the candidate state
            psi_of_candidate_state = self.calc_psi(candidate_state)

            # if the random number is less than |Ψ(candidate state)/Ψ(m-th state)|^2,
            # make the candidate state into the next state in the Markov chain
            if R < np.abs(psi_of_candidate_state / psi_of_m_th_state) ** 2:
                markov_chain[i] = candidate_state
                psi_of_m_th_state = psi_of_candidate_state
            # otherwise, make the (m-1)-th state into the next state in the Markov chain
            else:
                markov_chain[i] = markov_chain[i - 1].copy()

            for index, k in enumerate(markov_chain[i]):
                markov_chain_onehot[
                    i, index * self.n_max : self.n_max * (index + 1)
                ] = np.concatenate(
                    (
                        np.ones(int(k), dtype=np.int32),
                        np.zeros(self.n_max - int(k), dtype=np.int32),
                    ),
                    axis=None,
                )

        return markov_chain, markov_chain_onehot

    def calc_local_energy(self, input_state):
        """Calculates the local energy of the given state."""

        psi_of_input_state = self.calc_psi(input_state)
        local_energy = 0

        local_energy = local_energy + np.sum(
            0.5 * input_state * (input_state - 1) - self.mu * input_state
        )

        if self.t != 0:
            for row_index_of_matched_state in np.where(
                (self.hop_bras == input_state).all(axis=1)
            )[0]:
                psi_ket = self.calc_psi(self.hop_kets[row_index_of_matched_state])
                local_energy = (
                    local_energy
                    - self.t
                    * (psi_ket / psi_of_input_state)
                    * self.hop_values[row_index_of_matched_state]
                )

        return local_energy

    def calc_E(self, local_energies):
        """Calculates the variational energy and its variance from the given local energies."""

        E = np.mean(local_energies)

        var_E = (
            np.sum(local_energies ** 2) / self.P
            - np.sum(local_energies) ** 2 / self.P ** 2
        )

        return E, var_E

    def calc_derivatives_of_psi_wrt_network_params(self, samples):
        """Calculates the variational derivatives of the wavefunction with respect to the network parameters."""

        alpha_der = samples.copy()

        theta = np.tile(self.beta, (self.P, 1)) + (self.W @ samples.T).T

        beta_der = np.tanh(theta)

        W_der = np.empty((self.P, self.N_s * self.M * self.n_max), dtype=np.complex128)

        for i in np.arange(self.P):
            alpha_mesh_i, beta_mesh_i = np.meshgrid(alpha_der[i], beta_der[i])

            W_der[i, :] = (alpha_mesh_i * beta_mesh_i).flatten()

        return alpha_der, beta_der, W_der

    def calc_derivatives_of_E_wrt_network_params(self, samples, local_energies):
        """Calculates the derivatives of the variational energy with respect to the network parameters."""

        E, _ = self.calc_E(local_energies)

        dE_d_alpha = np.zeros(self.alpha.shape, dtype=np.complex128)
        dE_d_beta = np.zeros(self.beta.shape, dtype=np.complex128)
        dE_d_W = np.zeros(self.N_s * self.M * self.n_max, dtype=np.complex128)

        (
            alpha_der,
            beta_der,
            W_der,
        ) = self.calc_derivatives_of_psi_wrt_network_params(samples)

        for y in np.arange(self.N_s * self.n_max):
            O_alpha_y = alpha_der[:, y]
            s_ap_O_alpha_y_conj = np.mean(O_alpha_y.conj())
            dE_d_alpha[y] = (
                np.mean(local_energies * O_alpha_y.conj()) - E * s_ap_O_alpha_y_conj
            )

            for x in np.arange(self.M):
                O_beta_y = beta_der[:, x]
                s_ap_O_beta_y_conj = np.mean(O_beta_y.conj())
                dE_d_beta[x] = (
                    np.mean(local_energies * O_beta_y.conj()) - E * s_ap_O_beta_y_conj
                )

                z = y * self.M + x
                O_W_y = W_der[:, z]
                s_ap_O_W_y_conj = np.mean(O_W_y.conj())
                dE_d_W[z] = np.mean(local_energies * O_W_y.conj()) - E * s_ap_O_W_y_conj

        dE_d_W = np.reshape(dE_d_W, (self.M, self.N_s * self.n_max))

        return dE_d_alpha, dE_d_beta, dE_d_W

    def order_param(self, samples):
        """Calculates the order parameter."""

        probs = [np.abs(self.calc_psi(sample_i)) ** 2 for sample_i in samples]
        A = samples[:, 0] ** 2 * probs
        B = samples[:, 0] * probs

        return np.sum(A) / np.sum(probs) - (np.sum(B) / np.sum(probs)) ** 2

    def optimize(self):
        """Optimize the RBM ansatz."""

        E = []
        order_params = []

        s_alpha = np.zeros(self.N_s * self.n_max, dtype=np.complex128)
        s_beta = np.zeros(self.M, dtype=np.complex128)
        s_W = np.zeros((self.M, self.N_s * self.n_max), dtype=np.complex128)

        for p in np.arange(self.P):
            self.reset_psi_data()

            samples, samples_onehot = self.run_metropolis_hastings_sampler()

            local_energies = np.array(
                [self.calc_local_energy(sample_state_i) for sample_state_i in samples]
            )

            E_i, var_E_i = self.calc_E(local_energies)
            E.append(E_i)

            order_param_i = self.order_param(samples)
            order_params.append(order_param_i)

            (
                dE_d_alpha,
                dE_d_beta,
                dE_d_W,
            ) = self.calc_derivatives_of_E_wrt_network_params(
                samples_onehot, local_energies
            )

            s_alpha = self.gamma * s_alpha + (1 - self.gamma) * np.abs(dE_d_alpha) ** 2
            s_beta = self.gamma * s_beta + (1 - self.gamma) * np.abs(dE_d_beta) ** 2
            s_W = self.gamma * s_W + (1 - self.gamma) * np.abs(dE_d_W) ** 2

            self.alpha = (
                self.alpha
                - self.learning_rate / (np.sqrt(s_alpha) + self.epscut) * dE_d_alpha
            )
            self.beta = (
                self.beta
                - self.learning_rate / (np.sqrt(s_beta) + self.epscut) * dE_d_beta
            )
            self.W = self.W - self.learning_rate / (np.sqrt(s_W) + self.epscut) * dE_d_W

        return E, order_params


# result for mu = 1.5 & t = 0.35
t = 0.35
mu = 1.5
N_s = 4
M = 6
n_max = 4
learning_rate = 0.01
gamma = 0.9
epscut = 1e-8
P = 500

example = OneDimBHM_w_RBM(t, mu, N_s, M, n_max, learning_rate, gamma, epscut, P)

Es, order_params = example.optimize()

print("Energy: {}".format(Es[-1]))