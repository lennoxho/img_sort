#pragma once

#include <iostream>
#include <mutex>
#include <vector>

#include "boost/range/irange.hpp"
#include "boost/range/adaptor/filtered.hpp"
#include "boost/range/adaptor/transformed.hpp"

#define RUNTIME_ASSERT(cond) { if (!(cond)) throw std::runtime_error(#cond); }

namespace img_sort {

    class logger {
    public:
        enum type {
            info,
            warning,
            error,
            fatal
        };

        template <type T, typename... S>
        static void post(const S&... msg) {
            constexpr auto prefix = type_string<T>();

            std::ostream* os = &std::cout;
            if constexpr (T == fatal) {
                os = &std::cerr;
            }

            *os << prefix;
            (*os << ... << msg) << '\n';

            if constexpr (T == fatal) {
                os->flush();
            }
        }

    private:
        template <type T>
        static constexpr std::string_view type_string() {
            if constexpr (T == type::info) {
                return "Info        : ";
            }
            else if constexpr (T == type::warning) {
                return "Warning     : ";
            }
            else if constexpr (T == type::error) {
                return "Error       : ";
            }
            else if constexpr (T == type::fatal) {
                return "Fatal Error : ";
            }
            else {
                throw std::runtime_error("Unrecognised type enum!");
            }
        }

        std::mutex mtx;
    };

    template <typename T>
    class triangular_table {
        std::vector<T> m_data;
        std::size_t m_width;

        static constexpr std::size_t nth_triangular(std::size_t n) noexcept {
            return n * (n + 1) / 2;
        }

    public:
        using coordinate = std::pair<std::size_t, std::size_t>;

        class iterator {
            triangular_table* m_table = nullptr;
            std::size_t m_x = 0;
            std::size_t m_y = 1;
            std::size_t m_prev_triangular = 0;

        public:
            using difference_type = std::ptrdiff_t;
            using value_type = std::pair<coordinate, T&>;
            using pointer = value_type*;
            using reference = value_type;
            using iterator_category = std::forward_iterator_tag;

            iterator() {}

            iterator(triangular_table &table) 
                :m_table{ &table }
            {
                RUNTIME_ASSERT(m_table->m_width > 0);
            }

            reference operator*() const {
                RUNTIME_ASSERT(m_y < m_table->m_width);
                return { std::pair{ m_x, m_y }, m_table->m_data[m_prev_triangular + m_x] };
            }

            iterator &operator++() {
                ++m_x;
                if (m_x == m_y) {
                    m_prev_triangular += m_y;
                    ++m_y;
                    m_x = 0;
                }

                return *this;
            }

            iterator operator++(int) {
                auto tmp = *this;
                operator++();
                return tmp;
            }

            bool operator==(const iterator &other) const noexcept {
                if (m_table == nullptr && other.m_table == nullptr) {
                    return true;
                }
                else if (m_table == nullptr) {
                    return other.m_y == other.m_table->m_width;
                }
                else if (other.m_table == nullptr) {
                    return m_y == m_table->m_width;
                }
                else {
                    return 
                        m_table == other.m_table && 
                        m_x == other.m_x &&
                        m_y == other.m_y;
                }
            }

            bool operator!=(const iterator &other) const noexcept {
                return !operator==(other);
            }
        };

        triangular_table(std::size_t n, const T& val = T())
            :m_data( nth_triangular(n) - n, val ),
            m_width{ n }
        {
            RUNTIME_ASSERT(n > 0);
        }

        auto begin() { return iterator{ *this }; }
        auto end() { return iterator{}; }

        const T &operator()(std::size_t x, std::size_t y) const {
            RUNTIME_ASSERT(x < m_width && y < m_width);
            RUNTIME_ASSERT(x != y);

            if (y < x) std::swap(x, y);
            // y > x >= 0
            return m_data[nth_triangular(y - 1) + x];
        }

        auto row(std::size_t y) const {
            RUNTIME_ASSERT(y < m_width);
            auto func = [this, y](std::size_t x) {
                return std::pair{ std::pair{ x, y }, (*this)(x, y) };
            };

            return boost::irange<std::size_t>(0, m_width)
                | boost::adaptors::filtered([y](auto x) { return x != y; })
                | boost::adaptors::transformed(func);
        }
    };

}