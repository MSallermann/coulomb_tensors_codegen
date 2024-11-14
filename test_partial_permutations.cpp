#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>

/**
 * @brief Computes the binomial coefficient (N choose K) at compile time.
 *
 * This function uses recursion to calculate the number of ways to choose K elements
 * from a set of N elements without considering the order of selection.
 *
 * @param N The total number of items.
 * @param K The number of items to choose.
 * @return The binomial coefficient (N choose K).
 */
constexpr int binomial_coeff(int N, int K)
{
    return (K == 0 || K == N)
               ? 1
               : (N < K) ? 0 : binomial_coeff(N - 1, K - 1) + binomial_coeff(N - 1, K);
}

/**
 * @brief Recursively generates all valid count distributions for the first m elements.
 *
 * This function explores all possible ways to distribute the remaining_m counts
 * across the first k elements, respecting the maximum frequency of each element.
 *
 * @tparam k The number of distinct elements.
 * @tparam n_counts The maximum number of count distributions.
 * @param i The current index being processed.
 * @param counts_so_far The current count distribution being built.
 * @param remaining_m The remaining counts to distribute.
 * @param freq The frequency array representing the maximum allowed counts for each element.
 * @param counts_list The list where valid count distributions are stored.
 * @param counts_list_size The current size of the counts_list.
 */
template <size_t k, size_t n_counts>
void generate_counts(int                                       i,
                     std::array<int, k>&                       counts_so_far,
                     int                                       remaining_m,
                     const std::array<int, k>&                 freq,
                     std::array<std::array<int, k>, n_counts>& counts_list,
                     int&                                      counts_list_size)
{
    // Base case: All elements have been processed
    if (i == k)
    {
        // If all remaining counts have been distributed, store the count distribution
        if (remaining_m == 0)
        {
            counts_list[counts_list_size++] = counts_so_far;
        }
        return;
    }

    // Determine the maximum count for the current element
    int max_count = std::min(freq[i], remaining_m);
    for (int n_i = 0; n_i <= max_count; ++n_i)
    {
        // Assign a count to the current element
        counts_so_far[i] = n_i;

        // Recursively assign counts to the next element
        generate_counts<k>(
            i + 1, counts_so_far, remaining_m - n_i, freq, counts_list, counts_list_size);

        // Reset the count for the current element (optional for clarity)
        counts_so_far[i] = 0;
    }
}

/**
 * @brief Generates all unique permutations of a multiset and invokes a callback for each.
 *
 * This function constructs permutations by selecting elements based on their remaining counts.
 * It ensures that each permutation is unique by considering the multiplicity of elements.
 *
 * @tparam k The number of distinct elements in the multiset.
 * @tparam m The number of elements fixed in the first m positions.
 * @tparam n The total number of elements in the permutation.
 * @param remaining_counts An array representing the remaining counts of each element to be
 * permuted.
 * @param first_m_elements An array containing the first m elements that are fixed.
 * @param current_permutation An array used to build the current permutation.
 * @param current_size The current position in the permutation being constructed.
 * @param callback A function to be called with each complete permutation.
 */
template <size_t k, size_t m, size_t n, typename CallbackFunction>
void permute_multiset_with_callback(const std::array<int, k>& remaining_counts,
                                    const std::array<int, m>& first_m_elements,
                                    std::array<int, n - m>&   current_permutation,
                                    int                       current_size,
                                    CallbackFunction          callback)
{
    // Base case: A complete permutation has been constructed
    if (current_size == n - m)
    {
        // Combine the first m fixed elements with the current permutation
        std::array<int, n> full_permutation;
        for (size_t i = 0; i < m; ++i)
        {
            full_permutation[i] = first_m_elements[i];
        }
        for (size_t i = 0; i < n - m; ++i)
        {
            full_permutation[m + i] = current_permutation[i];
        }

        // Invoke the callback with the complete permutation
        callback(full_permutation);
        return;
    }

    // Iterate over all distinct elements
    for (size_t elem = 0; elem < k; ++elem)
    {
        if (remaining_counts[elem] > 0)
        {
            // Choose the current element by decrementing its remaining count
            std::array<int, k> new_remaining_counts = remaining_counts;
            new_remaining_counts[elem]--;

            // Add the current element to the permutation
            current_permutation[current_size] = elem;

            // Recursively build the rest of the permutation
            permute_multiset_with_callback<k, m, n>(new_remaining_counts,
                                                    first_m_elements,
                                                    current_permutation,
                                                    current_size + 1,
                                                    callback);
        }
    }
}

/**
 * @brief Generates all unique permutations of an array and executes a callback for each.
 *
 * This function handles duplicates by first sorting the input array and then generating
 * permutations in a way that avoids repeating identical sequences. It divides the permutation
 * process into fixing the first m elements and permuting the remaining elements.
 *
 * @tparam n The total number of elements in the input array.
 * @tparam m The number of elements to fix in the first m positions (sorted).
 * @tparam k The range of distinct elements (from 0 to k-1).
 * @param a The input array containing elements to permute.
 * @param callback A function to be called with each unique permutation.
 */
template <size_t n, size_t m, size_t k, typename CallbackFunction>
void generate_permutations(const std::array<int, n>& a, CallbackFunction callback)
{
    // Step 1: Sort the input array to handle duplicates effectively
    std::array<int, n> a_sorted = a;
    std::sort(a_sorted.begin(), a_sorted.end());

    // Step 2: Create a frequency array for elements 0 to k-1
    std::array<int, k> freq = {0};
    for (size_t i = 0; i < n; ++i)
    {
        int elem = a_sorted[i];
        freq[elem]++;
    }

    // Step 3: Calculate the maximum number of possible count distributions
    constexpr int max_count_distributions =
        binomial_coeff(m + k - 1, k - 1);  // (m + k -1 choose k -1)

    // Initialize a list to store all valid count distributions
    std::array<std::array<int, k>, max_count_distributions> counts_list = {};

    // Variable to keep track of the number of valid count distributions found
    int counts_list_size = 0;

    // Temporary array to build count distributions
    std::array<int, k> counts_so_far = {};

    // Generate all valid count distributions for the first m elements
    generate_counts<k>(0, counts_so_far, m, freq, counts_list, counts_list_size);

    // Step 4: For each count distribution, generate permutations of the remaining elements
    for (int idx = 0; idx < counts_list_size; ++idx)
    {
        const auto& count_distribution = counts_list[idx];

        // Construct the first m elements based on the current count distribution
        std::array<int, m> first_m_elements = {};
        size_t             first_m_size     = 0;
        for (size_t elem = 0; elem < k; ++elem)
        {
            for (int cnt = 0; cnt < count_distribution[elem]; ++cnt)
            {
                first_m_elements[first_m_size++] = static_cast<int>(elem);
            }
        }

        // Determine the remaining counts for elements after the first m
        std::array<int, k> remaining_counts = {};
        for (size_t elem = 0; elem < k; ++elem)
        {
            remaining_counts[elem] = freq[elem] - count_distribution[elem];
        }

        // Initialize an array to build the current permutation of the remaining elements
        std::array<int, n - m> current_permutation = {};

        // Generate all unique permutations of the remaining elements and invoke the callback
        permute_multiset_with_callback<k, m, n>(
            remaining_counts, first_m_elements, current_permutation, 0, callback);
    }
}

// ============================================================================
// Main Function: Example Usage
// ============================================================================
int main()
{
    constexpr size_t n = 9;  // Total number of elements
    constexpr size_t m = 5;  // Number of elements to be sorted in non-descending order
    constexpr size_t k = 3;  // Elements range from 0 to k-1 (i.e., 0, 1, 2)

    // Example input: elements are within [0, k-1] and can repeat
    std::array<int, n> a = {1, 1, 2, 2, 2, 1, 0, 0};

    // Vectors to store permutations from both methods
    std::vector<std::array<int, n>> generated_perms;
    std::vector<std::array<int, n>> brute_force_perms;

    // Callback function to collect generated permutations
    auto collect_generated_permutation = [&](const std::array<int, n>& perm) {
        generated_perms.emplace_back(perm);
    };

    // Generate permutations using the main implementation
    generate_permutations<n, m, k>(a, collect_generated_permutation);

    // --- Brute Force Approach ---
    // Step 1: Sort the input array for std::next_permutation
    std::array<int, n> a_sorted = a;
    std::sort(a_sorted.begin(), a_sorted.end());

    // Step 2: Generate all unique permutations using std::next_permutation
    do
    {
        // Check if the first m elements are in non-descending order
        bool valid = true;
        for (size_t i = 1; i < m; ++i)
        {
            if (a_sorted[i - 1] > a_sorted[i])
            {
                valid = false;
                break;
            }
        }
        if (valid)
        {
            brute_force_perms.emplace_back(a_sorted);
        }
    } while (std::next_permutation(a_sorted.begin(), a_sorted.end()));

    // --- Verification ---
    // Step 1: Check if both methods generated the same number of permutations
    if (generated_perms.size() != brute_force_perms.size())
    {
        std::cerr << "Test Failed: Different number of permutations generated.\n";
        std::cerr << "Generated: " << generated_perms.size()
                  << ", Brute Force: " << brute_force_perms.size() << "\n";
        return 1;
    }

    // Step 2: Sort both vectors for comparison
    auto sort_permutation = [&](const std::array<int, n>& a, const std::array<int, n>& b) -> bool {
        return a < b;
    };

    std::sort(generated_perms.begin(), generated_perms.end(), sort_permutation);
    std::sort(brute_force_perms.begin(), brute_force_perms.end(), sort_permutation);

    // Step 3: Compare each permutation
    bool test_passed = true;
    for (size_t i = 0; i < generated_perms.size(); ++i)
    {
        if (generated_perms[i] != brute_force_perms[i])
        {
            std::cerr << "Test Failed: Mismatch found at permutation index " << i << ".\n";
            std::cerr << "Generated: ";
            for (int num : generated_perms[i]) std::cout << num << ' ';
            std::cerr << "\nBrute Force: ";
            for (int num : brute_force_perms[i]) std::cout << num << ' ';
            std::cerr << '\n';
            test_passed = false;
            break;
        }
    }

    if (test_passed)
    {
        std::cout << "Test Passed: All permutations match the brute force approach.\n";
    }

    return 0;
}
