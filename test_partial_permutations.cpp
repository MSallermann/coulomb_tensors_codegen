#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>

// Function to generate all valid counts for the first m elements
template <size_t n>
void generate_counts(int                                  i,
                     std::array<int, n>&                  counts_so_far,
                     int                                  remaining_m,
                     const std::array<int, n>&            c_list,
                     int                                  k,
                     std::array<std::array<int, n>, 100>& counts_list,
                     int&                                 counts_list_size)
{
    if (i == k)
    {
        if (remaining_m == 0)
        {
            counts_list[counts_list_size++] = counts_so_far;
        }
        return;
    }
    int max_count = std::min(c_list[i], remaining_m);
    for (int n_i = 0; n_i <= max_count; ++n_i)
    {
        counts_so_far[i] = n_i;
        generate_counts(
            i + 1, counts_so_far, remaining_m - n_i, c_list, k, counts_list, counts_list_size);
        counts_so_far[i] = 0;  // Reset to zero (optional)
    }
}

// Function to generate unique permutations of the multiset without duplicates
template <size_t n, size_t m, typename CallbackFunction>
void permute_multiset(const std::array<int, n>& counts,
                      const std::array<int, n>& elements,
                      int                       k,
                      std::array<int, n - m>&   current_permutation,
                      int                       current_size,
                      CallbackFunction          callback)
{
    if (current_size == n - m)
    {
        // Convert current_permutation to a Permutation by appending it to the first m elements
        // This step will be handled in the main generation function
        // Hence, we pass the completed permutation to the callback there
        // Here, we'll assume the first m elements are already handled
        // and the callback will be called after combining
        return;
    }

    for (int i = 0; i < k; ++i)
    {
        if (counts[i] > 0)
        {
            // Skip duplicates: If the current element is the same as the previous and the previous
            // hasn't been used, skip
            if (i > 0 && elements[i] == elements[i - 1] && counts[i - 1] > counts[i])
            {
                continue;
            }
            std::array<int, n> new_counts = counts;
            new_counts[i]--;
            current_permutation[current_size] = elements[i];
            // Recursive call
            permute_multiset(
                new_counts, elements, k, current_permutation, current_size + 1, callback);
        }
    }
}

// Overloaded function to handle permutation generation and callback invocation
template <size_t n, size_t m, typename CallbackFunction>
void permute_multiset_with_callback(const std::array<int, n>& counts,
                                    const std::array<int, n>& elements,
                                    int                       k,
                                    std::array<int, n - m>&   current_permutation,
                                    int                       current_size,
                                    CallbackFunction          callback,
                                    const std::array<int, m>& first_m_elements)
{
    using Permutation = std::array<int, n>;

    if (current_size == n - m)
    {
        // Combine first m elements with the current permutation
        Permutation full_permutation;
        for (int i = 0; i < m; ++i)
        {
            full_permutation[i] = first_m_elements[i];
        }
        for (int i = 0; i < n - m; ++i)
        {
            full_permutation[m + i] = current_permutation[i];
        }
        // Invoke the callback with the complete permutation
        callback(full_permutation);
        return;
    }

    for (int i = 0; i < k; ++i)
    {
        if (counts[i] > 0)
        {
            // Skip duplicates: If the current element is the same as the previous and the previous
            // hasn't been used, skip
            if (i > 0 && elements[i] == elements[i - 1] && counts[i - 1] > counts[i])
            {
                continue;
            }
            std::array<int, n> new_counts = counts;
            new_counts[i]--;
            current_permutation[current_size] = elements[i];
            // Recursive call
            permute_multiset_with_callback(new_counts,
                                           elements,
                                           k,
                                           current_permutation,
                                           current_size + 1,
                                           callback,
                                           first_m_elements);
        }
    }
}

// Compile-time binomial coefficient function
constexpr int binomial_coeff(int N, int K)
{
    return (K == 0 || K == N)
               ? 1
               : (N < K) ? 0 : binomial_coeff(N - 1, K - 1) + binomial_coeff(N - 1, K);
}

// Main function to generate permutations and execute the callback
template <size_t n, size_t m, typename CallbackFunction>
void generate_permutations(const std::array<int, n>& a, CallbackFunction callback)
{
    // Step 1: Sort the input array
    std::array<int, n> a_sorted = a;
    std::sort(a_sorted.begin(), a_sorted.end());

    // Step 2: Identify unique elements and their counts
    std::array<int, n> elements = {};  // Unique elements
    std::array<int, n> counts   = {};  // Counts of each unique element
    int                k        = 0;   // Number of unique elements

    elements[0] = a_sorted[0];
    counts[0]   = 1;
    k           = 1;
    for (int i = 1; i < n; ++i)
    {
        if (a_sorted[i] == elements[k - 1])
        {
            counts[k - 1]++;
        }
        else
        {
            elements[k] = a_sorted[i];
            counts[k]   = 1;
            k++;
        }
    }

    std::cout << k << "\n";

    // Step 3: Generate all valid counts for the first m elements
    constexpr int max_count = 100;// binomial_coeff(m + k - 1, k - 1);

    std::array<std::array<int, n>, max_count> counts_list      = {};  // Adjust size as needed
    int                                 counts_list_size = 0;
    std::array<int, n>                  counts_so_far    = {};
    generate_counts(0, counts_so_far, m, counts, k, counts_list, counts_list_size);

    // Step 4: For each valid count distribution, generate permutations of the remaining elements
    for (int idx = 0; idx < counts_list_size; ++idx)
    {
        const auto&        n_list           = counts_list[idx];
        std::array<int, m> first_m_elements = {};
        int                first_m_size     = 0;
        std::array<int, n> remaining_counts = {};

        for (int idx2 = 0; idx2 < k; ++idx2)
        {
            int n_i = n_list[idx2];
            int e   = elements[idx2];

            for (int i = 0; i < n_i; ++i)
            {
                first_m_elements[first_m_size++] = e;
            }
            int remaining          = counts[idx2] - n_i;
            remaining_counts[idx2] = remaining;
        }

        // Ensure the first m elements are sorted
        // (They should already be sorted due to the way counts are generated)

        // Generate all unique permutations of the remaining elements and invoke the callback
        std::array<int, n - m> current_permutation = {};
        permute_multiset_with_callback(
            remaining_counts, elements, k, current_permutation, 0, callback, first_m_elements);
    }
}

int main()
{
    constexpr size_t n = 8;
    constexpr size_t m = 0;
    constexpr size_t k = 3;

    // Example input
    std::array<int, n> a = {1, 1, 2, 2, 2, 1, 0, 0};

    // Example callback function to print the permutation
    auto print_permutation = [&](const std::array<int, n>& perm) {
        for (int num : perm)
        {
            std::cout << num << ' ';
        }
        std::cout << '\n';
    };

    // Generate permutations and print them using the callback
    generate_permutations<n, m>(a, print_permutation);

    return 0;
}
