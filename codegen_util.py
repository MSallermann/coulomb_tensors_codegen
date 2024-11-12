from dataclasses import dataclass
import scipy as sp
import itertools
from pathlib import Path


def delta_recursion(alphabet: list, cur: set, n_delta_left: int, all_combinations: set):
    """Takes all combinations of two symbols from alphabet and adds them to a copy of the current combination.
    Then for every found combinations, removes the two symbols in the combination from the alphabet and recurses.
    If no kronecker deltas are left, adds the current combination to `all_combinations`.

    Args:
        alphabet (list): The list of available symbols
        cur (set): The current combination.
        n_delta (int): the number of delta functions that are left
        all_combinations (set): The set of index pairs under the delta function.
    """

    if n_delta_left >= 1:
        combinations = itertools.combinations(alphabet, 2)
        for p in combinations:
            cur_t = cur.copy()
            alphabet_t = alphabet.copy()
            alphabet_t.remove(p[0])
            alphabet_t.remove(p[1])
            cur_t.add(p)
            delta_recursion(
                alphabet_t.copy(),
                cur_t.copy(),
                n_delta_left=n_delta_left - 1,
                all_combinations=all_combinations,
            )
    else:
        for a in alphabet:
            cur.add(a)
        all_combinations.add(frozenset(cur))
        return None


def find_permutations(n_delta: int, n_r: int) -> set[frozenset[int]]:
    r"""
    Assume a tensor in Einstein notation with `n_delta` kronecker deltas and `n_r` position vectors.
    I.e `delta_{alpha beta} delta_{gamma delta} r_epsilon` would correspond to `n_delta = 2` and `n_r = 1`.
    This function finds all permuations of the indices that would lead to a new tensor.

    The type of the return value is a set of a frozenset since neither the order of `delta` functions and `r` (delta_{ab}r_c is equal to r_Cdelta_{ab}),
    nor the order of indices in the delta function matters (delta_{ab} is equal to delta_{ba})

    Args:
        n_delta (int): number of kronecker deltas
        n_r (int): number of position vectors

    Returns:
        permutations (set[frozenset[int]]): the permutations
    """

    n_symbols = 2 * n_delta + n_r
    alphabet = list(range(n_symbols))
    all_combinations = set()
    delta_recursion(alphabet, set(), n_delta, all_combinations)
    return all_combinations


@dataclass
class EinsteinTerm:
    r"""Represents a tensor in einstein summations of the form
        'prefactor/R^order * ( \delta_{ab} \delta_{cd} ... r_k r_l ... r_z + <permutations> )'
    with `n_delta` delta functions and `n_r` position vectors.
    The `permutations` refer to permutations of the ordered list of indices [abcde ...]
    """

    def __init__(self):
        self.permutations = set()
        self.prefactor = 0
        self.order = 0
        self.n_delta = 0
        self.n_r = 0


def T_Tensor(n: int) -> list[EinsteinTerm]:
    r"""
    Represents the `n`-th derivative of the 'T-Tensor', where T= 1/R = 1/sqrt(rx^2 + ry^2 + rz^2).
    For example at n=2, we have
        T_{ab} = \partial_{r_a} \partial_{r_b} 1/R.
    etc.

    Args:
        n (int): The order of the derivative

    Returns:
        list[EinsteinTerm]: result as a list of terms in einstein summation notation
    """

    einstein_terms = []

    if n % 2 == 0:
        lowest_order = n + 1
    else:
        lowest_order = n + 2

    highest_order = 2 * n + 1

    # print(f"{lowest_order = }")
    # print(f"{highest_order = }")

    # The overall 1/R^{exponent} scaling of the tensor
    r_scaling_exponent = n + 1
    # print(f"{r_scaling_exponent = }")

    for l in range(int((n + 1) / 2), n + 1):
        order = 2 * l + 1
        # print(order)

        # constant prefactor
        pref = (-1) ** l * sp.special.factorial2(2 * l - 1, exact=True)
        # print(f"{pref = }")

        n_r = order - r_scaling_exponent
        n_delta_functions = int((n - n_r) / 2)

        # print(f"{n_r = }")
        # print(f"{n_delta_functions = }")

        permutations = find_permutations(n_delta_functions, n_r)
        # print(permutations)

        term = EinsteinTerm()
        term.n_delta = n_delta_functions
        term.n_r = n_r
        term.prefactor = pref
        term.order = order
        term.permutations = permutations

        # print(term)

        einstein_terms.append(term)

    return einstein_terms


def insert_separator(items: list[str], sep=",") -> str:
    """Returns a string with `sep` between each item of the list
    e.g insert_separator(  ["ab", "c", "d", "efg"], sep=", " ) -> "ab, c, d, efg"
    """

    if len(items) == 0:
        return ""

    if len(items) == 1:
        return str(items[0])

    if len(items) == 2:
        return f"{items[0]}{sep}{items[1]}"

    res = ""
    res += str(items[0])
    res += sep

    for i in items[1:-1]:
        res += str(i)
        res += sep

    res += str(items[-1])
    return res


def sign(x):
    return "+" if x >= 0 else "-"


def unique_indices(rank: int) -> list:
    return list(itertools.combinations_with_replacement(range(3), rank))


def all_indices(rank: int) -> list:
    return list(itertools.product(range(3), repeat=rank))


def idx_to_unique_idx(indices: list[int]) -> list[int]:
    idx_new = list(indices)
    idx_new.sort()
    return idx_new


def tensor_type(rank):
    items = rank * ["3"]
    threes = insert_separator(items, ",")
    return f"Tensor<double, {threes}>"


def preamble(rank_t, rank_result):
    rank_sum = rank_t - rank_result

    res = f"""/**
* @brief Contract the rank {rank_t} Coulomb tensor with a rank {rank_sum} tensor Q.
*
* @tparam SW_Func_T
* @param r position difference vector
* @param sw_func switching function with signature sw_func(double, int) -> double
* @return Tensor<double> (a rank {rank_result} tensor)
*/
template <typename SW_Func_T>
    """

    items = rank_t * ["3"]
    threes = insert_separator(items, ",")

    res += f"inline {tensor_type(rank_result)} contract_Coulomb_{rank_t}_{rank_result}(const Tensor<double, 3>& r, const SW_Func_T& sw_func, const {tensor_type(rank_sum)} & Q)"
    res += "\n{\n"

    res += "    const double R1 = norm(r);\n"
    res += "    const double R2 = R1*R1;\n"
    max_order = 2 * rank_t + 1

    if rank_t % 2 == 0:
        min_order = rank_t + 1
    else:
        min_order = rank_t + 2

    for o in range(3, max_order + 2, 2):
        res += f"    const double R{o} = R{o-2} * R2;\n"

    for o in range(min_order, max_order + 2, 2):
        res += f"    const double SW{o} = sw_func(R1, {o});\n"

    return res


def precompute_r_component_powers(n: int):
    res = ""

    if n % 2 == 0:
        lowest_order = n + 1
    else:
        lowest_order = n + 2

    highest_order = 2 * n + 1

    res += f"    const double rx{0} = 1.0;\n"
    res += f"    const double ry{0} = 1.0;\n"
    res += f"    const double rz{0} = 1.0;\n"

    for pow in range(1, highest_order - lowest_order + 2):
        res += f"    const double rx{pow} = rx{pow-1} * r[0];\n"
        res += f"    const double ry{pow} = ry{pow-1} * r[1];\n"
        res += f"    const double rz{pow} = rz{pow-1} * r[2];\n"

    return res


def filter_list(my_list):
    filtered_list = []
    for item in my_list:
        if item not in filtered_list:
            filtered_list.append(item)
    return filtered_list


def sum_repeated_elements(my_list):
    filtered_list = filter_list(my_list)

    for item in filtered_list:
        factor = my_list.count(item)

        idx = filtered_list.index(item)
        filtered_list[idx] *= factor

    return filtered_list
