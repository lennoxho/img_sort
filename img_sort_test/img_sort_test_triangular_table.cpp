#include "../img_sort/img_sort.h"
#include "catch.hpp"

TEST_CASE("size 0", "[triagular_table]") {
    CHECK_THROWS(img_sort::triangular_table<int>{ 0 });
}

TEST_CASE("size 1", "[triagular_table]") {
    img_sort::triangular_table<int> table{ 1, -1 };
    CHECK(std::distance(table.begin(), table.end()) == 0);
    CHECK(boost::distance(table.row(0)) == 0);
}

TEST_CASE("size 2", "[triagular_table]") {
    using table_type = img_sort::triangular_table<int>;
    table_type table{ 2, -1 };

    CHECK(table(0, 1) == -1);
    CHECK(table(1, 0) == -1);
    CHECK(std::distance(table.begin(), table.end()) == 1);

    using it_value_type = std::pair<typename table_type::coordinate, int>;

    GIVEN("unmodified") {
        SECTION("iterator") {
            std::vector<it_value_type> expected = {
                { {0, 1}, -1 }
            };

            std::vector<it_value_type> actual{ table.begin(), table.end() };
            CHECK(expected == actual);
        }

        SECTION("row unmodified") {
            std::vector<it_value_type> expected = {
                { {1, 0}, -1 }
            };

            auto r = table.row(0);
            std::vector<it_value_type> actual{ r.begin(), r.end() };
            CHECK(expected == actual);

            expected = {
                { {0, 1}, -1 }
            };

            r = table.row(1);
            actual.assign(r.begin(), r.end());
            CHECK(expected == actual);
        }
    }

    GIVEN("modified") {
        (*table.begin()).second = 42;
        CHECK(table(0, 1) == 42);
        CHECK(table(1, 0) == 42);

        SECTION("iterator") {
            std::vector<it_value_type> expected = {
                { {0, 1}, 42 }
            };

            std::vector<it_value_type> actual{ table.begin(), table.end() };
            CHECK(expected == actual);
        }

        SECTION("row unmodified") {
            std::vector<it_value_type> expected = {
                { {1, 0}, 42 }
            };

            auto r = table.row(0);
            std::vector<it_value_type> actual{ r.begin(), r.end() };
            CHECK(expected == actual);

            expected = {
                { {0, 1}, 42 }
            };

            r = table.row(1);
            actual.assign(r.begin(), r.end());
            CHECK(expected == actual);
        }
    }
}

TEST_CASE("size 4", "[triagular_table]") {
    using table_type = img_sort::triangular_table<int>;
    table_type table{ 4, -1 };

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (i != j) CHECK(table(i, j) == -1);
        }
    }

    CHECK(std::distance(table.begin(), table.end()) == 6);

    using it_value_type = std::pair<typename table_type::coordinate, int>;

    GIVEN("unmodified") {
        SECTION("iterator") {
            std::vector<it_value_type> expected = {
                { {0, 1}, -1 },
                { {0, 2}, -1 },
                { {1, 2}, -1 },
                { {0, 3}, -1 },
                { {1, 3}, -1 },
                { {2, 3}, -1 },
            };
            std::vector<it_value_type> actual{ table.begin(), table.end() };
            CHECK(expected == actual);
        }

        SECTION("row unmodified") {
            for (std::size_t i = 0; i < 4; ++i) {
                CHECK(boost::distance(table.row(i)) == 3);

                auto r = table.row(i);
                auto it = r.begin();
                for (std::size_t j = 0; j < 4; ++j) {
                    if (i != j) {
                        auto [coord, v] = *it;
                        CHECK(coord.first == j);
                        CHECK(coord.second == i);
                        CHECK(v == -1);
                        ++it;
                    }
                }
            }
        }
    }

    GIVEN("modified") {
        for (auto p : table) {
            auto [x, y] = p.first;
            p.second = static_cast<int>(y * 10 + x);
        }

        CHECK(table(0, 1) == 10);
        CHECK(table(1, 2) == 21);

        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                if (j < i) {
                    CHECK(table(i, j) == table(j, i));
                }
                else if (i > j) {
                    CHECK(table(i, j) == (j * 10 + i));
                }
            }
        }

        SECTION("iterator") {
            for (auto p : table) {
                auto [x, y] = p.first;
                CHECK(p.second == (y * 10 + x));
            }
        }

        SECTION("row unmodified") {
            for (std::size_t i = 0; i < 4; ++i) {
                CHECK(boost::distance(table.row(i)) == 3);

                for (auto p : table.row(i)) {
                    auto [x, y] = p.first;
                    if (y < x) {
                        CHECK(table(x, y) == table(y, x));
                    }
                    else if (x > y) {
                        CHECK(table(x, y) == (y * 10 + x));
                    }
                }
            }
        }
    }
}