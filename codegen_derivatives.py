from pathlib import Path
from codegen_script import *

alphabet = [chr(97 + i) for i in range(24)]


def function_signature(rank_pairs, ranks_results_unique, ranks_sum_unique):

    res = ""
    res += "void multi_Coulomb_contract("

    # , const {tensor_type(rank_sum)} & Q, const double te)"

    function_arguments = ["const Tensor<double, 3>& r"]

    for rs in ranks_sum_unique:
        function_arguments.append(f"const {tensor_type(rs)} & Q{rs}")

    for rr in ranks_results_unique:
        function_arguments.append(f"{tensor_type(rr)} & res{rr}")

    function_arguments.append("double te")

    res += insert_separator(function_arguments, ",")

    res += ")"

    res += "\n{\n"  # open function body

    return res


def precompute_r_component_powers(rank_min, rank_max):
    res = ""
    res += "    const double R1 = norm(r);\n"
    res += "    const double R2 = R1*R1;\n"
    max_order = 2 * rank_max + 1

    if rank_min % 2 == 0:
        min_order = rank_min + 1
    else:
        min_order = rank_min + 2

    for o in range(3, max_order + 2, 2):
        res += f"    const double R{o} = R{o-2} * R2;\n"

    for o in range(min_order, max_order + 2, 2):
        res += f"    const double SW{o} = stonedampingNF(R1, {o}, te);\n"

    res += f"    const double rx{0} = 1.0;\n"
    res += f"    const double ry{0} = 1.0;\n"
    res += f"    const double rz{0} = 1.0;\n"

    for pow in range(1, max_order - min_order + 2):
        res += f"    const double rx{pow} = rx{pow-1} * r[0];\n"
        res += f"    const double ry{pow} = ry{pow-1} * r[1];\n"
        res += f"    const double rz{pow} = rz{pow-1} * r[2];\n"

    return res


def insert_callback(rank_tensor: int, ranks_results: list[int]):
    res = "\n\n"

    max_rank_result = max(ranks_results)
    min_rank_result = min(ranks_results)

    res += f"auto cb = [&]( const std::array<int,{rank_tensor}> & idx )\n"
    res += "{\n"

    for rr in ranks_results:
        rank_sum = rank_tensor - rr
        idx_lhs = insert_separator([f"idx[{i}]" for i in range(rr)])
        idx_rhs = insert_separator([f"idx[{i}]" for i in range(rr, rank_tensor)])
        # res += f"    res_T{rank_tensor}_R{rr}( { idx_lhs } ) += t_cur_{rank_tensor} * Q{rank_sum}({idx_rhs});\n"
        res += f"    res{rr}( { idx_lhs } ) += t_cur_{rank_tensor} * Q{rank_sum}({idx_rhs});\n"

    res += "};\n"

    res += f"std::array<int, {rank_tensor}> cur_idx = {{ {insert_separator(alphabet[:rank_tensor])} }};\n"

    res += (
        f"generate_permutations<{rank_tensor}, {min_rank_result}, 3>(cur_idx, cb);\n\n"
    )

    return res


def write_cpp_function(rank_pairs):
    res = ""

    n_rank_pairs = len(rank_pairs)
    max_rank = rank_pairs[n_rank_pairs - 1][0]
    min_rank = rank_pairs[0][0]

    terms = []

    # do some accounting first
    rank_tensors = []

    ranks_results_unique = []
    ranks_sum_unique = []

    rank_results_from_tensor = [[] for i in range(max_rank + 1)]
    rank_tensors_from_result = [[] for i in range(max_rank + 1)]
    for i, (rank_tensor, ranks_results) in enumerate(rank_pairs):
        rank_tensors.append(rank_tensor)
        rank_results_from_tensor[rank_tensor] = ranks_results
        for rr in ranks_results:
            if rr not in ranks_results_unique:
                ranks_results_unique.append(rr)

            if rank_tensor - rr not in ranks_sum_unique:
                ranks_sum_unique.append(rank_tensor - rr)
            rank_tensors_from_result[rr].append(rank_tensor)

    ranks_results_unique.sort()
    ranks_sum_unique.sort()

    print(rank_results_from_tensor)

    res += function_signature(rank_pairs, ranks_results_unique, ranks_sum_unique)

    res += precompute_r_component_powers(min_rank, max_rank)

    # First write all the unique components
    for rank_tensor, rank_result in rank_pairs:
        t = T_Tensor(rank_tensor)
        terms.append(t)
        res += write_out_unique_t_tensor_components_arr(
            t, rank_tensor, f"T{rank_tensor}"
        )
        res += "\n\n"
    pass

    # Write temporary variables for results
    # for rank_tensor, ranks_results in rank_pairs:
    #     for r_res in ranks_results:
    #         res += f"     {tensor_type(r_res)} res_T{rank_tensor}_R{r_res}(0.0);\n"
    # pass

    # Write variables for counters
    for rank in rank_tensors:
        res += f"int count_T{rank} = 0;\n"

    for iloop in range(max_rank):
        start = 0 if iloop == 0 else alphabet[iloop - 1]
        a = alphabet[iloop]
        res += f"for(int {a} = {start}; {a} < 3; {a}++)\n"
        res += "{\n"

        rank = iloop + 1

        # insert the temporary results
        for r_t in rank_tensors_from_result[rank]:
            res += f"     double tmp_T{r_t}_R{rank}(0.0);\n"

        if rank in rank_tensors:
            res += f"const double t_cur_{rank} = T{rank}[count_T{rank}];\n"
            res += f"count_T{rank}++;\n"

            res += insert_callback(rank, rank_results_from_tensor[rank])

    for iloop in range(max_rank):
        res += "}\n"

    res += "}\n"  # close function
    return res


if __name__ == "__main__":
    output_path = Path("./output")

    rank_pairs = [
        [4, [3]],
        [5, [3, 4]],
        [6, [3, 4, 5]],
        [7, [3, 4, 5]],
        [8, [4, 5]],
        [9, [5]],
    ]

    with open(output_path / "coulomb_multi_contraction.cpp", "w") as f:
        f.write(
            """
    #include "tensor.hpp"
    #include "stonedamping.hpp"
    #include <algorithm>
    #include "coulomb_tensor_contraction.hpp"
    #include "coulomb_tensor_utils.hpp"
    #include "permute_multiset.hpp"

    namespace SCME::Coulomb_Tensors
    {
        """
        )

        f.write(write_cpp_function(rank_pairs))
        f.write("\n\n")
        f.write("}")
