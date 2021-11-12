#pragma once

#include "model.h"

namespace estimator::solver {
    // This is the estimator class for estimating a homography matrix between two images.
    // A model estimation method and error calculation method are implemented
    class SolverEngine {
    public:
        SolverEngine() = default;

        ~SolverEngine() = default;

        // The minimum number of points required for the estimation
        static constexpr size_t sampleSize() {
            return 0;
        }

        // The maximum number of solutions returned by the estimator
        static constexpr size_t maximumSolutions() {
            return 1;
        }

        // Determines if there is a chance of returning multiple models
        // the function 'estimateModel' is applied.
        static constexpr bool returnMultipleModels() {
            return maximumSolutions() > 1;
        }

//        // Estimate the model parameters from the given point sample using weighted fitting if possible.
//        bool estimateModel(
//                const cv::Mat &data_,
//                const size_t *sample_,
//                size_t sample_number_,
//                std::vector<Model> &models_,
//                const double *weights_ = nullptr);
    };
}