from codegen_util import *
import numpy as np
from dataclasses import dataclass
from typing import Self


@dataclass(slots=True)
class PermutationExplicit(object):
    r_factors: list[int]
    prefactor: int = 1.0

    def __init__(self):
        # r_factors is a list with three entries, representing the contribution from one permutation.
        # the tuple (p1,p2,p3) represents a term of the form rx^p1 * ry^p2 * rz^p3
        # to find the contribution from the current einsteint term, the contributions
        # from all perturbations are added up
        self.r_factors: list[int] = np.zeros(3, dtype=int)
        self.prefactor = 1.0

    def __add__(self, val: Self):
        new = PermutationExplicit()
        new.r_factors = self.r_factors
        new.prefactor = self.prefactor + val.prefactor
        return new

    def __mul__(self, factor):
        new = PermutationExplicit()
        new.r_factors = self.r_factors
        new.prefactor = self.prefactor * factor
        return new

    def __repr__(self) -> str:
        return (
            f"<Permutation: prefactor = {self.prefactor}, r_factors = {self.r_factors}>"
        )

    def __str__(self) -> str:
        items = []

        if self.r_factors[0] > 0:
            items.append(
                f"rx{self.r_factors[0]}"
            )

        if self.r_factors[1] > 0:
            items.append(
                f"ry{self.r_factors[1]}"
            )

        if self.r_factors[2] > 0:
            items.append(
                f"rz{self.r_factors[2]}"
            )

        if len(items) == 0:
            return f"{self.prefactor}"

        return f"{self.prefactor} * " + insert_separator(items, " * ")

    # equality operator based on r_factors
    def __eq__(self, val: Self) -> np.bool:
        assert isinstance(val, PermutationExplicit)

        if np.all(self.r_factors == val.r_factors):
            return True
        else:
            return False


def evaluate_component(indices: list[int], terms: list[EinsteinTerm]):
    contributions_terms = []

    for t in terms:
        # print(t)

        # The contributions from the current term (sum up later)
        summands_permutations: list[PermutationExplicit] = []

        # We have to sum up the contribution over all permutations
        for p in t.permutations:
            pexp = PermutationExplicit()

            # contributes is a bool that tells us if the permutation contributes or not (duh)
            # it gets set to false if any of the delta's has unequal indices
            contributes = True

            for i in p:
                # If delta function, we check if the indices are different
                # If yes, the permutation does not contribute and we just continue
                if not isinstance(i, int):
                    if indices[i[0]] != indices[i[1]]:
                        contributes = False
                        pexp.prefactor = 0.0
                        break
                else:
                    # increment the r_factor
                    pexp.r_factors[indices[i]] += 1

            # if the permutation does indeed contribute, we append it to the contributions_term list
            if contributes:
                summands_permutations.append(pexp)

        # [print(s) for s in summands_permutations] 
        # [print(s) for s in sum_repeated_elements(summands_permutations)] 

        # sum up all the repeated elements before appending
        contributions_terms.append( sum_repeated_elements(summands_permutations) )

    comp = ""

    # Now we build up the components
    for t, cont in zip(terms, contributions_terms):

        if len(cont) == 0:
            continue

        items = []
        for pexp in cont:
            items.append(str(pexp))

        comp += (
            f" {sign(t.prefactor)} {abs(t.prefactor):.1f} * SW{t.order} / R{t.order} * ("
            + insert_separator(items, " + ")
            + ")"
        )

    return comp

def write_out_unique_t_tensor_components( terms : list[EinsteinTerm], rank ):
    res = ""
    for indices in unique_indices(rank):
        lhs = f"t_{insert_separator(indices,'')}" 
        rhs = evaluate_component(indices, terms)
        res += f"    const double {lhs} = {rhs};\n"

    return res


def write_out_unique_t_tensor_components_arr( terms : list[EinsteinTerm], rank ):
    res = ""

    idx_list = unique_indices(rank)

    res += f"    std::array<double, {len(idx_list)}> T" + "{\n       "

    components = []
    for indices in idx_list:
        rhs = evaluate_component(indices, terms)
        components.append(f"{rhs}")

    res += insert_separator(components, ",\n       ")

    res += "\n    };\n"

    return res

def write_tensor_contraction( rank_result, rank_t, target_symbol = "res", input_symbol = "Q" ):
    res = ""

    rank_sum = rank_t - rank_result

    # add the variable we will write the result to
    res += f"    {tensor_type(rank_result)} {target_symbol}" +"{};\n\n" 

    for indices_result in all_indices(rank_result):
        lhs = f"{target_symbol}({insert_separator(indices_result,',')})"

        summands = []
        for indices_sum in all_indices(rank_sum):
            indices_t = indices_result + indices_sum
            indices_t = idx_to_unique_idx(indices_t)
            summands.append(  f"t_{insert_separator(indices_t, "")} * {input_symbol}({insert_separator(indices_sum, ',')})" )

        rhs = insert_separator(summands, "+")
        res += f"    {lhs} = {rhs};\n"

    return res

def write_tensor_contraction_loops( rank_result, rank_t, target_symbol = "res", input_symbol = "Q" ):
    res = ""

    alphabet = [chr(97+i) for i in range(rank_t)]

    rank_sum = rank_t - rank_result

    # add the variable we will write the result to
    res += f"    {tensor_type(rank_result)} {target_symbol}" +"(0.0);\n\n" 

    # Write the head of the tensor loops
    for i in range(rank_t):
        a = alphabet[i]
        res += f"for(int {a} = 0; {a} < 3; {a}++)\n"
        res += "{\n"


    res += f"const int idx_t = Util::get_position_in_lookup_array<{rank_t}>( {{ {insert_separator(alphabet)} }} );\n"
    res += f"res( {insert_separator( alphabet[:rank_result] )} ) += T[idx_t] * Q({insert_separator( alphabet[rank_result:] )});\n"

    for i in range(rank_t):
        res += "}\n"

    return res

def write_tensor_contraction_loops_heaps_algorithm( rank_result, rank_t, target_symbol = "res", input_symbol = "Q" ):
    res = ""

    alphabet = [chr(97+i) for i in range(rank_t)]

    rank_sum = rank_t - rank_result

    # add the variable we will write the result to
    res += f"    {tensor_type(rank_result)} {target_symbol}" +"(0.0);\n\n" 

    # Write the head of the tensor loops
    res += "int count{0};\n"

    for i in range(rank_t):
        start = 0 if i==0 else alphabet[i-1]
        a = alphabet[i]
        res += f"for(int {a} = {start}; {a} < 3; {a}++)\n"
        res += "{\n"


    res += f"const double t_cur = T[count];\n"

    idx_lhs = insert_separator( [f"idx[{i}]" for i in range(rank_result)])
    idx_rhs = insert_separator( [f"idx[{i}]" for i in range(rank_result, rank_t)])

    res += f"auto cb = [&]( const std::array<int,{rank_t}> & idx )\n"
    res += "{\n"
    res += f"    res( { idx_lhs } ) += t_cur * Q({idx_rhs});\n"
    res += "};\n"

    res += f"std::array<int, {rank_t}> A = {{ {insert_separator(alphabet)} }};\n"

    res += f"Util::generate_permutations<{rank_t}>({rank_t}, A, cb);\n"

    res+= "count++;\n"

    for i in range(rank_t):
        res += "}\n"

    return res


def write_tensor_contraction_loops_permutations_v2( rank_result, rank_t, target_symbol = "res", input_symbol = "Q" ):
    res = ""

    alphabet = [chr(97+i) for i in range(rank_t)]

    rank_sum = rank_t - rank_result

    # add the variable we will write the result to
    res += f"    {tensor_type(rank_result)} {target_symbol}" +"(0.0);\n\n" 

    # Write the head of the tensor loops
    # res += "int count{0};\n"

    for i in range(rank_result):
        start = 0 if i==0 else alphabet[i-1]
        a = alphabet[i]
        res += f"for(int {a} = {start}; {a} < 3; {a}++)\n"
        res += "{\n"

    res += "\ndouble tmp{0.0};\n\n"

    # res += f"const double mult = Util::multiplicity<{rank_result}>( {{{insert_separator(alphabet[:rank_result])}}} );\n"

    for i in range(rank_result, rank_t):
        start = 0
        a = alphabet[i]
        res += f"for(int {a} = {start}; {a} < 3; {a}++)\n"
        res += "{\n"

    res += f"const int count = Util::get_position_in_lookup_array<{rank_t}>( {{ {insert_separator(alphabet)} }} );\n"
    res += f"const double t_cur = T[count];\n"

    # res += f"std::array<int, {rank_sum}> idx_q = {{ {insert_separator(alphabet[rank_result:])} }};\n"

    # idx_rhs = insert_separator( [f"idx_q[{i}]" for i in range(rank_sum)])
    # res += " do {\n"
    # res += f"   tmp += mult * t_cur * Q({idx_rhs});\n"
    res += f"   tmp +=t_cur * Q({ insert_separator(alphabet[rank_result:]) });\n"

    # res += "} while (std::next_permutation(idx_q.begin(), idx_q.end()));\n\n"

    # res+= "count++;\n"

    for i in range(rank_sum):
        res += "}\n"


    res += f"std::array<int, {rank_result}> idx_r = {{ {insert_separator(alphabet[:rank_result])} }};\n"
    idx_lhs = insert_separator( [f"idx_r[{i}]" for i in range(rank_result)])
    res += " do {\n"
    res += f"   res( { idx_lhs } ) = tmp;\n"
    res += "} while (std::next_permutation(idx_r.begin(), idx_r.end()));\n\n"



    for i in range(rank_sum,rank_t):
        res += "}\n"

    return res


def write_tensor_contraction_loops_permutations_v3( rank_result, rank_t, target_symbol = "res", input_symbol = "Q" ):
    res = ""

    alphabet = [chr(97+i) for i in range(rank_t)]

    rank_sum = rank_t - rank_result

    # add the variable we will write the result to
    res += f"    {tensor_type(rank_result)} {target_symbol}" +"(0.0);\n\n" 

    # Write the head of the tensor loops
    res += "int count{0};\n"

    for i in range(rank_result):
        start = 0 if i==0 else alphabet[i-1]
        a = alphabet[i]
        res += f"for(int {a} = {start}; {a} < 3; {a}++)\n"
        res += "{\n"

    res += "\ndouble tmp{0.0};\n\n"

    # res += f"const double mult = Util::multiplicity<{rank_result}>( {{{insert_separator(alphabet[:rank_result])}}} );\n"

    for i in range(rank_result, rank_t):
        start = 0 if i==0 else alphabet[i-1]
        a = alphabet[i]
        res += f"for(int {a} = {start}; {a} < 3; {a}++)\n"
        res += "{\n"

    # res += f"const int count = Util::get_position_in_lookup_array<{rank_t}>( {{ {insert_separator(alphabet)} }} );\n"
    res += f"const double t_cur = T[count];\n"

    res += f"std::array<int, {rank_sum}> idx_q = {{ {insert_separator(alphabet[rank_result:])} }};\n"

    idx_rhs = insert_separator( [f"idx_q[{i}]" for i in range(rank_sum)])
    res += " do {\n"
    res += f"   tmp += t_cur * Q({idx_rhs});\n"
    # res += f"   tmp +=t_cur * Q({ insert_separator(alphabet[rank_result:]) });\n"

    res += "} while (std::next_permutation(idx_q.begin(), idx_q.end()));\n\n"

    res+= "count++;\n"

    for i in range(rank_sum):
        res += "}\n"


    res += f"std::array<int, {rank_result}> idx_r = {{ {insert_separator(alphabet[:rank_result])} }};\n"
    idx_lhs = insert_separator( [f"idx_r[{i}]" for i in range(rank_result)])
    res += " do {\n"
    res += f"   res( { idx_lhs } ) = tmp;\n"
    res += "} while (std::next_permutation(idx_r.begin(), idx_r.end()));\n\n"



    for i in range(rank_sum,rank_t):
        res += "}\n"

    return res

def write_tensor_contraction_loops_permutations( rank_result, rank_t, target_symbol = "res", input_symbol = "Q" ):
    res = ""

    alphabet = [chr(97+i) for i in range(rank_t)]

    rank_sum = rank_t - rank_result

    # add the variable we will write the result to
    res += f"    {tensor_type(rank_result)} {target_symbol}" +"(0.0);\n\n" 

    # Write the head of the tensor loops
    res += "int count{0};\n"

    for i in range(rank_t):
        start = 0 if i==0 else alphabet[i-1]
        a = alphabet[i]
        res += f"for(int {a} = {start}; {a} < 3; {a}++)\n"
        res += "{\n"

    res += f"const double t_cur = T[count];\n"

    idx_lhs = insert_separator( [f"idx[{i}]" for i in range(rank_result)])
    idx_rhs = insert_separator( [f"idx[{i}]" for i in range(rank_result, rank_t)])

    res += f"std::array<int, {rank_t}> idx = {{ {insert_separator(alphabet)} }};\n"

    res += " do {\n"
    res += f"    res( { idx_lhs } ) += t_cur * Q({idx_rhs});\n"
    res += "} while (std::next_permutation(idx.begin(), idx.end()));\n"

    res+= "count++;\n"

    for i in range(rank_t):
        res += "}\n"

    return res

def write_cpp_function(rank_tensor, rank_result):
    res = preamble(rank_tensor, rank_result)
    res += precompute_r_component_powers(rank_tensor)

    res += "\n"

    terms = T_Tensor(rank_tensor)
    # res += write_out_unique_t_tensor_components(terms, rank_tensor)
    res += write_out_unique_t_tensor_components_arr(terms, rank_tensor)

    res += "\n"

    # res += write_tensor_contraction(rank_result, rank_tensor)
    # res += write_tensor_contraction_loops(rank_result, rank_tensor)
    # res += write_tensor_contraction_loops_heaps_algorithm(rank_result, rank_tensor)
    # res += write_tensor_contraction_loops_permutations(rank_result, rank_tensor)
    # res += write_tensor_contraction_loops_permutations_v2(rank_result, rank_tensor)
    res += write_tensor_contraction_loops_permutations_v3(rank_result, rank_tensor)

    res += "\n"

    res += "    return res;\n"

    res += "}\n"

    return res

if __name__ == "__main__":
    comp = evaluate_component([0,0,1,1], T_Tensor(4))
    print(comp)
    # exit(0)

    output_path = Path("./output")
    output_path.mkdir(exist_ok=True)

    rank_pairs = [
        [3, 2],
        [4, 2],
        [5, 2],
        [6, 2],
        [4, 3],
        [5, 3],
        [6, 3],
        [7, 3],
        [5, 4],
        [6, 4],
        [7, 4],
        [8, 4],
        [6, 5],
        [7, 5],
        [8, 5],
        [9, 5],
    ]


    with open(output_path / "coulomb_tensor_contraction.cpp", "w" ) as f:
        f.write("""
#include "tensor.hpp"
#include "stonedamping.hpp"
#include <algorithm>
#include "coulomb_tensor_contraction.hpp"
#include "coulomb_tensor_utils.hpp"

namespace SCME::Coulomb_Tensors
{
    """)
        for rank_tensor, rank_target in rank_pairs:
            f.write(write_cpp_function(rank_tensor, rank_target))
            f.write("\n\n")
        f.write("}")


    with open(output_path / "coulomb_tensor_contraction.hpp", "w" ) as f:
        f.write("""
#pragma once
#include "tensor.hpp"

namespace SCME::Coulomb_Tensors
{
""")
        for rank_tensor, rank_target in rank_pairs:
            f.write(wrapped_header(rank_tensor, rank_target))
            f.write("\n\n")
        f.write("}")


#     with open(output_path / "coulomb_tensor_contraction.cpp", "w" ) as f:
#         f.write("""
# #include "tensor.hpp"
# #include "coulomb_tensor_contraction.hpp"
# #include "generic_coulomb_tensor_contraction.hpp"

# namespace SCME::Coulomb_Tensors
# {
# """)
#         for rank_tensor, rank_target in rank_pairs:
#             f.write(wrapped_implementation(rank_tensor, rank_target))
#             f.write("\n\n")
#         f.write("}")

