#pragma once

#include <random>
#include <algorithm>

namespace utils {
    class UniformRandomGenerator {
    protected:
        std::mt19937 generator;
        std::uniform_int_distribution<size_t> generate;

    public:
        UniformRandomGenerator() {
            std::random_device rand_dev;
            generator = std::mt19937(rand_dev());
        }

        ~UniformRandomGenerator() {

        }

        inline int getRandomNumber() {
            return generate(generator);
        }

        inline void resetGenerator(size_t min_range_, size_t max_range_) {
            generate = std::uniform_int_distribution<size_t>(min_range_, max_range_);
        }

        inline void generateUniqueRandomSet(size_t *sample_, size_t sample_size_) {
            for (size_t i = 0; i < sample_size_; i++) {
                sample_[i] = generate(generator);
                for (int j = i - 1; j >= 0; j--) {
                    if (sample_[i] == sample_[j]) {
                        i--;
                        break;
                    }
                }
            }
        }

        inline void generateUniqueRandomSet(size_t *sample_, size_t sample_size_, size_t max_) {
            resetGenerator(0, max_);
            for (size_t i = 0; i < sample_size_; i++) {
                sample_[i] = generate(generator);
                for (int j = i - 1; j >= 0; j--) {
                    if (sample_[i] == sample_[j]) {
                        i--;
                        break;
                    }
                }
            }
        }

        inline void generateUniqueRandomSet(size_t *sample_, size_t sample_size_, size_t max_, size_t to_skip_) {
            resetGenerator(0, max_);
            for (size_t i = 0; i < sample_size_; i++) {
                sample_[i] = generate(generator);
                if (sample_[i] == to_skip_) {
                    i--;
                    continue;
                }

                for (int j = i - 1; j >= 0; j--) {
                    if (sample_[i] == sample_[j]) {
                        i--;
                        break;
                    }
                }
            }
        }
    };
}
