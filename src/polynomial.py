import torch
from torch import Tensor, nn


def _prepare_input(x: Tensor, n_features: int) -> Tensor:
    """
    Validate the input tensor and convert a single sample
    of shape [n_features] into a batch of shape [1, n_features].
    """

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(
            "x must have shape [batch_size, n_features]"
        )

    if x.shape[1] != n_features:
        raise ValueError(
            f"Expected {n_features} features, got {x.shape[1]}"
        )

    return x


class PolynomialRegression2(nn.Module):
    """
    Second-order polynomial regression.

    The model is defined as:

        y = w0
            + sum_i w_i * x_i
            + sum_{i < j} w_ij * x_i * x_j

    Every pair of features has its own independent
    interaction coefficient w_ij.
    """

    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()

        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError(
                "n_features must be a positive integer"
            )

        self.n_features = n_features

        # Global bias w0.
        self.bias = nn.Parameter(
            torch.zeros(())
        )

        # Linear coefficients w_i.
        self.linear_weights = nn.Parameter(
            torch.zeros(n_features)
        )

        # Generate indices for all feature pairs i < j.
        pair_indices = torch.triu_indices(
            n_features,
            n_features,
            offset=1,
        )

        # Store the first index of every pair.
        self.register_buffer(
            "pair_i",
            pair_indices[0],
        )

        # Store the second index of every pair.
        self.register_buffer(
            "pair_j",
            pair_indices[1],
        )

        n_pairs = pair_indices.shape[1]

        # Independent coefficient w_ij for every feature pair.
        self.pair_weights = nn.Parameter(
            torch.zeros(n_pairs)
        )

    def linear_part(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute the linear part:

            sum_i w_i * x_i
        """

        x = _prepare_input(
            x,
            self.n_features,
        )

        return x @ self.linear_weights

    def interaction(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute all second-order polynomial interactions:

            sum_{i < j} w_ij * x_i * x_j
        """

        x = _prepare_input(
            x,
            self.n_features,
        )

        # Compute x_i * x_j for every feature pair.
        pair_products = (
            x[:, self.pair_i]
            * x[:, self.pair_j]
        )

        # Multiply every pair product by its own coefficient w_ij
        # and sum all interaction terms.
        return pair_products @ self.pair_weights

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute the complete polynomial regression prediction.
        """

        x = _prepare_input(
            x,
            self.n_features,
        )

        return (
            self.bias
            + self.linear_part(x)
            + self.interaction(x)
        )


class FactorizedPolynomialRegression(nn.Module):
    """
    Second-order factorized polynomial regression.

    Instead of learning an independent interaction coefficient
    w_ij for every pair of features, each feature gets a latent
    vector v_i.

    The interaction coefficient is represented as:

        w_ij = <v_i, v_j>

    Therefore the model becomes:

        y = w0
            + sum_i w_i * x_i
            + sum_{i < j} <v_i, v_j> * x_i * x_j

    This is the main factorization idea used in
    second-order Factorization Machines.
    """

    def __init__(
        self,
        n_features: int,
        n_factors: int = 16,
        init_std: float = 0.01,
        fast_interaction: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError(
                "n_features must be a positive integer"
            )

        if not isinstance(n_factors, int) or n_factors <= 0:
            raise ValueError(
                "n_factors must be a positive integer"
            )

        if init_std <= 0:
            raise ValueError(
                "init_std must be positive"
            )

        self.n_features = n_features
        self.n_factors = n_factors
        self.init_std = float(init_std)
        self.fast_interaction = fast_interaction

        # Global bias w0.
        self.bias = nn.Parameter(
            torch.zeros(())
        )

        # Linear coefficients w_i.
        self.linear_weights = nn.Parameter(
            torch.zeros(n_features)
        )

        # Latent factor matrix V.
        #
        # Each feature i has its own latent vector v_i
        # of size n_factors.
        #
        # Shape:
        # [n_features, n_factors]
        self.factors = nn.Parameter(
            torch.empty(
                n_features,
                n_factors,
            )
        )

        self._reset_factor_parameters()

        # Generate indices for all feature pairs i < j.
        pair_indices = torch.triu_indices(
            n_features,
            n_features,
            offset=1,
        )

        self.register_buffer(
            "pair_i",
            pair_indices[0],
        )

        self.register_buffer(
            "pair_j",
            pair_indices[1],
        )

    def _reset_factor_parameters(
        self,
    ) -> None:
        """
        Initialize latent factors using a normal distribution.
        """

        nn.init.normal_(
            self.factors,
            mean=0.0,
            std=self.init_std,
        )

    def reset_parameters(
        self,
    ) -> None:
        """
        Reset all model parameters.
        """

        with torch.no_grad():
            self.bias.zero_()
            self.linear_weights.zero_()

        self._reset_factor_parameters()

    def linear_part(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute the linear part:

            sum_i w_i * x_i
        """

        x = _prepare_input(
            x,
            self.n_features,
        )

        return x @ self.linear_weights

    def interaction_direct(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute factorized interactions directly:

            sum_{i < j}
                <v_i, v_j> * x_i * x_j

        This implementation explicitly considers every pair
        of features.

        Time complexity:

            O(k * n^2)

        where:
            n = number of features
            k = number of latent factors
        """

        x = _prepare_input(
            x,
            self.n_features,
        )

        # Get latent vectors v_i for the left side
        # of every feature pair.
        left_factors = self.factors[
            self.pair_i
        ]

        # Get latent vectors v_j for the right side
        # of every feature pair.
        right_factors = self.factors[
            self.pair_j
        ]

        # Compute the dot product:
        #
        # <v_i, v_j>
        #
        # for every feature pair.
        pair_coefficients = (
            left_factors
            * right_factors
        ).sum(dim=1)

        # Compute:
        #
        # x_i * x_j
        #
        # for every sample and every feature pair.
        pair_products = (
            x[:, self.pair_i]
            * x[:, self.pair_j]
        )

        # Compute:
        #
        # sum_{i < j}
        # <v_i, v_j> * x_i * x_j
        return (
            pair_products
            * pair_coefficients
        ).sum(dim=1)

    def interaction_fast(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute factorized interactions using the
        Factorization Machine sum-of-squares identity.

        Instead of explicitly calculating every feature pair:

            sum_{i < j}
                <v_i, v_j> * x_i * x_j

        we use:

            1/2 * sum_f [
                (sum_i v_if * x_i)^2
                -
                sum_i v_if^2 * x_i^2
            ]

        Time complexity:

            O(k * n)

        instead of:

            O(k * n^2)

        This is the computational trick that makes
        Factorization Machines efficient.
        """

        x = _prepare_input(
            x,
            self.n_features,
        )

        # Compute for every latent factor f:
        #
        # sum_i v_if * x_i
        #
        # Result shape:
        # [batch_size, n_factors]
        factor_sums = (
            x @ self.factors
        )

        # Compute:
        #
        # (sum_i v_if * x_i)^2
        squared_factor_sums = (
            factor_sums.square()
        )

        # Compute:
        #
        # sum_i v_if^2 * x_i^2
        #
        # for every latent factor.
        factor_square_sums = (
            x.square()
            @ self.factors.square()
        )

        # Apply the FM sum-of-squares identity.
        return 0.5 * (
            squared_factor_sums
            - factor_square_sums
        ).sum(dim=1)

    def interaction(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Choose between the fast and direct
        interaction implementations.
        """

        if self.fast_interaction:
            return self.interaction_fast(x)

        return self.interaction_direct(x)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute the complete factorized polynomial
        regression prediction.
        """

        x = _prepare_input(
            x,
            self.n_features,
        )

        return (
            self.bias
            + self.linear_part(x)
            + self.interaction(x)
        )

    def pair_coefficient(
        self,
        i: int,
        j: int,
    ) -> Tensor:
        """
        Return the effective polynomial interaction
        coefficient between features i and j:

            w_ij = <v_i, v_j>
        """

        if i < 0 or i >= self.n_features:
            raise IndexError(
                "feature index i is out of range"
            )

        if j < 0 or j >= self.n_features:
            raise IndexError(
                "feature index j is out of range"
            )

        if i == j:
            raise ValueError(
                "second-order FM interactions use i != j"
            )

        return torch.dot(
            self.factors[i],
            self.factors[j],
        )

    def interaction_parameter_count(
        self,
    ) -> int:
        """
        Return the number of parameters used to represent
        pairwise interactions in the factorized model.

        The factorized representation uses:

            n_features * n_factors

        parameters instead of approximately:

            n_features^2 / 2

        independent polynomial interaction coefficients.
        """

        return (
            self.n_features
            * self.n_factors
        )
