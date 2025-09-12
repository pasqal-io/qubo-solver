from __future__ import annotations

import random
from typing import Mapping, Optional, Sequence, Union

import networkx as nx
import numpy as np
from pulser.register.base_register import BaseRegister

VariableId = Union[str, int]


class Qubo:
    """Simple container class for a Qubo problem

    This class provides some convenience methods for
    creating a Qubo to and from a matrix or from a dictionary of terms
    and coefficients

    Attributes:
        terms (list[tuple[VariableId, VariableId]]): a set of non-zero Qubo
            terms. This is a list
            of 2-dimensional indices corresponding to the binary variables
            appearing in the Qubo
        coeffs (list[float]): the coefficients in front of each term of the Qubo
    """

    def __init__(
        self,
        terms: Sequence[tuple[VariableId, VariableId]],
        coeffs: list[float],
        register: Optional[BaseRegister] = None,
    ):
        self._terms = list(terms)
        self._coeffs = coeffs
        self._register = register

    @property
    def terms(self) -> list[tuple[VariableId, VariableId]]:
        return self._terms

    @property
    def coeffs(self) -> list[float]:
        return self._coeffs

    def as_matrix(self, variables: Optional[list[VariableId]] = None) -> np.ndarray:
        """Return the dense upper triangular matrix corresponding to the Qubo instance

        The matrix is triangular and constructed in such a way that the row
        and column indices of each element correspond to the Qubo term and the
        element value is the coefficients. For instance:

        a = [[1, -2, -3], [0, -1, -1], [0, 0, -3]]

        corresponds to the following Qubo:

        Q(x) = x0 -2*x0x1 -x1 -3*x0x2 -x1x2 -3*x2

        Notice that the resulting matrix is always upper triangular.

        Args:
            variables: If present, the returned matrix will follow the same
                order as the variable ids in this list. There can even be
                variable ids that are not used in the terms, in which case it
                will pad the matrix with zeros. If absent, it will follow the
                order given by the `variables()` method.

        Returns:
            np.ndarray: the upper triangular matrix corresponding to
                the Qubo problem
        """

        if variables is None:
            variables = self.variables()
        size = len(variables)
        matrix = np.zeros((size, size))

        for terms, value in zip(self.terms, self.coeffs):
            positions = [variables.index(term) for term in terms]
            matrix[min(positions)][max(positions)] += value

        return np.triu(matrix)

    def as_graph(self) -> nx.Graph:
        graph = nx.Graph()

        for (term0, term1), coeff in zip(self.terms, self.coeffs):
            if term0 == term1:
                graph.add_node(term0, weight=coeff)
            else:
                graph.add_edge(term0, term1, weight=coeff)

        return graph

    def variables(self) -> list[VariableId]:
        """
        Returns:
            List of all the variables' names that are used in the Qubo
                formulation. It is sorted by ascending order of integers, then
                alphabetically for strings.
        """

        result = set()
        for u, v in self.terms:
            result.add(u)
            result.add(v)

        return sorted(list(result), key=lambda x: (not isinstance(x, int), x))

    def node_coeffs(self, include_absent: bool = False) -> dict[VariableId, float]:
        """

        Args:
            include_absent: If `True`, include in the dictionary the variables
                that don't have any linear coefficients. In that case, the
                coefficient 0 will apply.

        Returns:
            A dictionary where the keys are the variables' names, and the
                values their linear coefficients. Note that it excludes
                quadratic terms.
        """

        coeffs = {
            term0: coeff for (term0, term1), coeff in zip(self.terms, self.coeffs) if term0 == term1
        }

        if include_absent:
            for term in self.variables():
                coeffs[term] = coeffs.get(term, 0)

        return coeffs

    def evaluate(self, values: Mapping[VariableId, float]) -> float:
        """

        Args:
            values: Values to attribute to the variables.

        Returns:
            Result of the Qubo formula given the provided values to the
                variables.
        """

        values = dict(values)
        result = 0.0
        for (term0, term1), coeff in zip(self.terms, self.coeffs):
            result += coeff * values.get(term0, 0) * values.get(term1, 0)
        return result

    @staticmethod
    def from_matrix(matrix: np.ndarray | list) -> Qubo:
        """Return a QUBO instance from a matrix representation

        The matrix must be 2-dimensional and either symmetric or traingular.
        The matrix is constructed in such a way that the row and column indices
        of each element correspond to the term and the element value is
        the coefficients. For instance, the following QUBO problem:

        Q(x) = x0 -2*x0x1 -x1 -3*x0x2 -x1x2 -3*x2

        corresponds to the following matrix:

        a = [[1, -2, -3], [0, -1, -1], [0, 0, -3]]

        Args:
            matrix (np.ndarray | list): the input matrix representing the QUBO
                constructed as explained above

        Returns:
            Qubo: the corresponding QUBO instance
        """
        if isinstance(matrix, list):
            matrix = np.array(matrix)

        is_valid = (
            len(matrix.shape) == 2
            and matrix.shape[0] == matrix.shape[1]
            and (
                np.allclose(matrix, matrix.T)
                or np.allclose(matrix, np.triu(matrix))
                or np.allclose(matrix, np.tril(matrix))
            )
        )

        if not is_valid:
            raise ValueError(
                "The input matrix shall be a 2-dimension square matrix "
                "either symmetric or triangular"
            )

        matrix = tri_lower_to_upper(matrix)
        # make sure the matrix is upper triangular
        # to avoid duplicates when generating the terms
        if np.allclose(matrix, matrix.T):
            matrix = np.triu(matrix)

        terms = []
        coeffs = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = float(matrix[i][j])
                if value != 0.0:
                    terms.append((i, j))
                    coeffs.append(value)

        return Qubo(terms=terms, coeffs=coeffs)

    @staticmethod
    def from_terms(terms: Mapping[tuple[VariableId, VariableId] | VariableId, float]) -> Qubo:
        """Generate a QUBO instance or a matrix from a set of terms

        Terms are given as a dictionary with the following form:

        terms = {
            0: -1,
            (1, ): -3,
            (0, 1): 2
        }

        where the key is either an integer or a 1-tuple corresponding to
        a single term, or a 2-tuple with the cross-terms indices. For instance,
        the above dictionary corresponds to the following QUBO:

        Q(x) = -x0 -3*x1 + 2*x0x1

        Args:
            terms (list[tuple[int, ...]]): list of terms

        Returns:
            Qubo: the generate QUBO instance
        """
        _coeffs = []
        _terms = []
        for key, value in terms.items():

            if isinstance(key, (int, str)):
                term = key, key
            elif isinstance(key, tuple) and len(key) == 1:
                term = key[0], key[0]
            else:
                term = key

            _terms.append(term)
            _coeffs.append(value)

        return Qubo(terms=_terms, coeffs=_coeffs)

    @staticmethod
    def from_graph(graph: nx.Graph) -> Qubo:
        """Construct a Qubo problem from a graph, possibly having
        "weight" attributes on its nodes and edges."""

        terms = dict()

        for node, data in graph.nodes.data():
            terms[node] = float(data.get("weight", 0))

        for n1, n2, weight in graph.edges.data("weight"):
            if weight is None:
                raise ValueError(f"edge {(n1, n2)} has no weight attribute")
            terms[n1, n2] = float(weight)

        return Qubo.from_terms(terms)


def tri_lower_to_upper(matrix: np.ndarray) -> np.ndarray:
    if np.allclose(matrix, np.tril(matrix)):
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    return matrix


def random_qubo(size: int, density: float = 1.0, bounds: tuple[int, int] = (-10, 10)) -> Qubo:
    """Generates a NxN symetric matrix with uniform random coefficients between
    lower_bound (default = -10) and upper_bound (default = 10) given a certain
    density of the matrix determined by the density argument (default = 1.0)

    Args:
        size (int): the first dimension of the square QUBO matrix
        density (float, optional): determine density of the resulting QUBO matrix
        bounds (tuple[int, int], optional): bounds for the values of the coefficients

    Returns:
        Qubo: the generated QUBO instance
    """

    matrix = np.zeros((size, size))

    for i in range(size):
        matrix[i][i] = random.uniform(*bounds)
        for j in range(i + 1, size):
            if random.random() <= density:
                matrix[i][j] = random.uniform(*bounds)

    return Qubo.from_matrix(matrix)
