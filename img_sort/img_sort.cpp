// Template matching
// 1. Resize each image to the same size, stretch to fit
// 2. Perform template matching between each pair of resized images


#include "img_sort.h"

#include <algorithm>
#include <execution>
#include <filesystem>
#include <string>
#include <string_view>
#include <optional>
#include <stack>
#include <queue>

#include "boost/container/small_vector.hpp"
#include "boost/format.hpp"
#include "boost/range/adaptor/reversed.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace img_sort {

    static constexpr auto execution_policy = std::execution::par;

    struct histogram {
        cv::Mat mat;
        std::filesystem::path filename;

        histogram() {}

        histogram(cv::Mat &&m, const std::filesystem::path &f)
        :mat{ std::move(m) },
         filename{ f }
        {}

        void clear() {
            mat.release();
        }
    };

    class tree {
#ifndef NDEBUG
        std::vector<bool> m_parent;
#endif
        std::vector<boost::container::small_vector<std::size_t, 1>> m_adjacency_list;
        std::size_t m_num_edges = 0;

    public:
        tree(std::size_t size)
            :m_adjacency_list(size)
        {
            RUNTIME_ASSERT(size > 0);
#ifndef NDEBUG
            m_parent.assign(size, false);
#endif
        }

        bool try_insert(std::size_t parent, std::size_t child) {
            RUNTIME_ASSERT(parent < m_adjacency_list.size());
            
#ifndef NDEBUG
            RUNTIME_ASSERT(child < m_parent.size());
            if (contains(child)) {
                return false;
            }
            m_parent[child] = true;
#endif
            m_adjacency_list[parent].emplace_back(child);
            ++m_num_edges;
            return true;
        }

#ifndef NDEBUG
        bool contains(std::size_t node) const {
            return node == 0 || m_parent[node];
        }
#endif

        auto &children(std::size_t node) const {
            RUNTIME_ASSERT(node < m_adjacency_list.size());
            return m_adjacency_list[node];
        }

        std::size_t num_edges() const noexcept {
            return m_num_edges;
        }
    };

    cv::Mat calculate_histogram(const std::filesystem::path &filename) {
        try {
            cv::Mat img = cv::imread(filename.string());
            if (img.empty()) {
                logger::post<logger::warning>("Failed to load ", filename);
                return {};
            }

            cv::Mat hist;

            int bbins = 32, gbins = 32, rbins = 32;
            int histSize[] = { bbins, gbins, rbins };

            float branges[] = { 0, 256 };
            float granges[] = { 0, 256 };
            float tranges[] = { 0, 256 };

            const float* ranges[] = { branges, granges, tranges };
            int channels[] = { 0, 1, 2 };

            cv::calcHist(&img, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);

            return hist;
        }
        catch (...) {
            logger::post<logger::error>("Failed to calculate histogram for ", filename);
        }

        return {};
    }

    double compute_histogram_diff(const histogram &lhs, const histogram &rhs) {
        return cv::compareHist(lhs.mat, rhs.mat, cv::HISTCMP_BHATTACHARYYA);
        //return cv::compareHist(lhs.mat, rhs.mat, cv::HISTCMP_HELLINGER);
        //return cv::compareHist(lhs.mat, rhs.mat, cv::HISTCMP_KL_DIV);
    }

    template <typename T>
    tree compute_mst(std::size_t size, const triangular_table<T> &weights) {
        RUNTIME_ASSERT(size >= 2);
        tree t{ size };

        struct pq_entry {
            std::size_t source;
            std::size_t destination;
            T cost = std::numeric_limits<T>::max();
        };

        std::vector<pq_entry> candidates(size);
        for (std::size_t i = 0; i < size; ++i) {
            candidates[i].destination = i;
        }

        const pq_entry dummy_entry;
        const auto final_num_edges = size - 1;
        std::size_t just_inserted_index = 0;
        std::size_t just_inserted = 0;

        while (t.num_edges() < final_num_edges) {
            std::swap(candidates[just_inserted_index], candidates[candidates.size() - 1]);
            candidates.pop_back();

            const pq_entry* min_entry = &dummy_entry;

            // Might be interesting to parallelise for large size
            for (auto &curr_candidate : candidates) {
                const auto cost_to_just_inserted = weights(just_inserted, curr_candidate.destination);
                if (cost_to_just_inserted <= curr_candidate.cost) {
                    curr_candidate.source = just_inserted;
                    curr_candidate.cost = cost_to_just_inserted;
                }

                if (cost_to_just_inserted <= min_entry->cost) {
                    min_entry = &curr_candidate;
                }
            }

            bool insert_result = t.try_insert(min_entry->source, min_entry->destination);
            RUNTIME_ASSERT(insert_result);
            just_inserted_index = min_entry - candidates.data();
            just_inserted = min_entry->destination;
        }

        return t;
    }

    std::optional<std::vector<std::size_t>> pre_order(const tree &mst) {
        std::vector<std::size_t> order;
        order.reserve(mst.num_edges() + 1);

        std::stack<std::size_t, std::vector<std::size_t>> stack;
        stack.push(0);

        while (!stack.empty()) {
            std::size_t curr_node = stack.top();
            stack.pop();

            order.push_back(curr_node);

            for (std::size_t child : boost::adaptors::reverse(mst.children(curr_node))) {
                stack.push(child);
            }
        }

        RUNTIME_ASSERT(order.size() == (mst.num_edges() + 1));
        return order;
    }
}

int main(int argc, const char** argv) {
    using logger = img_sort::logger;

    if (argc != 3) {
        logger::post<logger::error>("Usage: img_sort <source directory> <output directory>");
        return -1;
    }

    const auto source_directory = std::filesystem::path{ argv[1] };
    const auto output_directory = std::filesystem::path{ argv[2] };
    if (std::filesystem::equivalent(source_directory, output_directory)) {
        logger::post<logger::error>("Source and destination directories and equivalent!");
        return -1;
    }

    //
    // listdir
    //

    if (!std::filesystem::is_directory(source_directory)) {
        logger::post<logger::error>(source_directory, " is not a directory");
        return -1;
    }

    logger::post<logger::info>("Searching for images in ", source_directory, "...");

    std::vector<std::filesystem::path> filenames;
    {
        auto is_recognised_img_extension = [](const auto &entry) {
            const auto ext = entry.path().extension().string();
            return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".jfif";
        };

        const auto range_of_img_files =
            boost::make_iterator_range( std::filesystem::directory_iterator{ source_directory,
                                                                             std::filesystem::directory_options::follow_directory_symlink },
                                        std::filesystem::directory_iterator{} )
            | boost::adaptors::filtered(is_recognised_img_extension)
            | boost::adaptors::filtered([](const auto &entry) { return entry.is_regular_file() || entry.is_symlink(); })
            | boost::adaptors::transformed([](const auto &entry) { return entry.path(); });

        filenames.assign(range_of_img_files.begin(), range_of_img_files.end());
    }

    if (filenames.empty()) {
        logger::post<logger::info>(source_directory, " is empty. Nothing to do");
        return 0;
    }

    //
    // Read images from disk and compute histograms
    //

    logger::post<logger::info>("Found ", filenames.size(), " images. Computing histograms...");

    std::vector<img_sort::histogram> histograms(filenames.size());
    {
        logger::benchmark([&]() {
            std::transform(img_sort::execution_policy, filenames.begin(), filenames.end(), histograms.begin(),
                [](const auto &f) { return img_sort::histogram{ img_sort::calculate_histogram(f), f }; });
        });

        auto new_end = std::partition(histograms.begin(), histograms.end(), [](const auto &h) { return !h.mat.empty(); });
        histograms.erase(new_end, histograms.end());
    }

    if (histograms.empty()) {
        logger::post<logger::warning>("No histograms were computed");
        return -1;
    }
    else if (histograms.size() == 1) {
        logger::post<logger::info>("Only one image loaded. Nothing to do");
        return 0;
    }

    //
    // Calculate differences
    //

    logger::post<logger::info>("Computed ", histograms.size(), " histograms. Calculating differences...");

    img_sort::triangular_table<double> diff_table{ histograms.size() };
    {
        auto compute_diff = [&](auto kvp) {
            auto [coord, res] = kvp;
            auto [x, y] = coord;

            RUNTIME_ASSERT(x != y);
            res = img_sort::compute_histogram_diff(histograms[x], histograms[y]);
        };

        logger::benchmark([&]() { std::for_each(img_sort::execution_policy, diff_table.begin(), diff_table.end(), compute_diff); });
        // Reduce memory footprint
        std::for_each(img_sort::execution_policy, histograms.begin(), histograms.end(), [](auto &h) { h.clear(); });
    }

    //
    // Create MST
    //

    logger::post<logger::info>("Computing MST...");
    const img_sort::tree mst = logger::benchmark([&]() { return img_sort::compute_mst(histograms.size(), diff_table); });

    // Reduce memory footprint
    diff_table.clear();

    //
    // Perform traversal
    //

    logger::post<logger::info>("Generating sort order...");
    const auto sort_order_opt = logger::benchmark([&]() { return img_sort::pre_order(mst); });
    if (!sort_order_opt) return -1;

    //
    // Create symlinks in output directory
    //

    logger::post<logger::info>("Populating output directory ", output_directory, "...");
    std::filesystem::create_directories(output_directory);

    std::size_t idx = 0;
    for (std::size_t entry : *sort_order_opt) {
        const auto src_path = histograms[entry].filename;

        std::filesystem::path dest_name = (boost::format{ "%05zu." } % idx++).str();
        dest_name += src_path.filename();
        
        std::filesystem::create_hard_link(src_path, output_directory / dest_name);
    }
}