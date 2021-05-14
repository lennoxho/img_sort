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
        static constexpr auto no_parent = std::numeric_limits<std::size_t>::max();

        std::vector<std::size_t> m_parent;
        std::vector<boost::container::small_vector<std::size_t, 1>> m_adjacency_list;
        std::size_t m_num_edges = 0;

    public:
        tree(std::size_t size)
            :m_parent(size, no_parent),
            m_adjacency_list(size)
        {
            RUNTIME_ASSERT(size > 0);
        }

        bool try_insert(std::size_t parent, std::size_t child) {
            RUNTIME_ASSERT(parent < m_adjacency_list.size());
            RUNTIME_ASSERT(child < m_parent.size());

            if (contains(child)) {
                return false;
            }

            m_parent[child] = parent;
            m_adjacency_list[parent].emplace_back(child);
            ++m_num_edges;
            return true;
        }

        bool contains(std::size_t node) const {
            return node == 0 || m_parent[node] != no_parent;
        }

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
        tree t{ size };
        
        struct pq_entry {
            std::size_t source;
            std::size_t destination;
            T weight;

            pq_entry(std::size_t src, std::size_t dst, T w)
                :source{ src },
                destination{ dst },
                weight{ w }
            {}

            bool operator<(const pq_entry &other) const noexcept {
                return weight > other.weight;
            }
        };

        std::priority_queue<pq_entry> heap;
        for (const auto &p : weights.row(0)) {
            auto [dst, src] = p.first;
            heap.emplace(src, dst, p.second);
        }

        while (t.num_edges() + 1 < size) {
            pq_entry curr = heap.top();
            heap.pop();

            if (t.try_insert(curr.source, curr.destination)) {
                for (const auto &p : weights.row(curr.destination)) {
                    auto [dst, src] = p.first;
                    if (!t.contains(dst)) {
                        heap.emplace(src, dst, p.second);
                    }
                }
            }
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

    //
    // listdir
    //

    const auto source_directory = std::filesystem::path{ argv[1] };
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
        std::transform(img_sort::execution_policy, filenames.begin(), filenames.end(), histograms.begin(),
            [](const auto &f) { return img_sort::histogram{ img_sort::calculate_histogram(f), f }; });

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

        std::for_each(img_sort::execution_policy, diff_table.begin(), diff_table.end(), compute_diff);
        std::for_each(img_sort::execution_policy, histograms.begin(), histograms.end(), [](auto &h) { h.clear(); });
    }

    //
    // Create MST
    //

    logger::post<logger::info>("Computing MST...");
    const img_sort::tree mst = img_sort::compute_mst(histograms.size(), diff_table);

    //
    // Perform traversal
    //

    logger::post<logger::info>("Generating sort order...");
    const auto sort_order_opt = img_sort::pre_order(mst);
    if (!sort_order_opt) return -1;

    //
    // Create symlinks in output directory
    //

    const auto output_directory = std::filesystem::path{ argv[2] };
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