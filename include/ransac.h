#pragma once

#include "model.h"
#include "model_score.h"
#include "samplers/sampler.h"
#include "gamma_values.cpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

template<class ModelEstimator>
class RANSAC {
public:
    explicit RANSAC() :
            iteration_limit(std::numeric_limits<size_t>::max()),
            maximum_threshold(10.0),
            minimum_iteration_number(50),
            interrupting_threshold(1.0),
            last_iteration_number(0),
            log_confidence(0),
            point_number(0),
            RANSAC_version(std::move(RANSAC_version_)) {}

    ~RANSAC() = default;

    /*!
     * A function to run RANSAC.
     * @param points_ The input data points
     * @param confidence_ The required confidence in the results
     * @param estimator_ The model estimator
     * @param sampler_ The sampler used
     * @param obtained_model_ The estimated model parameters
     * @param iteration_number_ The number of iterations done
     * @param model_score_ The score of the estimated model
     * @return whether succeed
     */
    bool run(const cv::Mat &points_, double confidence_, ModelEstimator &estimator_, sampler::Sampler<cv::Mat, size_t> &sampler_,
             Model &obtained_model_, int &iteration_number_, ModelScore &model_score_);

protected:
    size_t iteration_limit; // Maximum number of iterations allowed
    size_t minimum_iteration_number; // Minimum number of iteration before terminating
    double maximum_threshold; // The maximum sigma value
    size_t core_number; // Number of core used in sigma-consensus
    int point_number; // The current point number
    int last_iteration_number; // The iteration number implied by the last run of sigma-consensus
    double log_confidence; // The logarithm of the required confidence
    size_t partition_number; // Number of partitions used to speed up sigma-consensus
    double interrupting_threshold; // A threshold to speed up RANSAC by interrupting the sigma-consensus procedure whenever there is no chance of being better than the previous so-far-the-best model

};

template<class ModelEstimator>
bool RANSAC<ModelEstimator>::run(
        const cv::Mat &points_,
        const double confidence_,
        ModelEstimator &estimator_,
        sampler::Sampler<cv::Mat, size_t> &sampler_,
        Model &obtained_model_,
        int &iteration_number_,
        ModelScore &model_score_) {
    // Initialize variables
    log_confidence = log(1.0 - confidence_); // The logarithm of 1 - confidence
    point_number = points_.rows; // Number of points
    const size_t sample_size = estimator_.sampleSize(); // The sample size required for the estimation
    size_t max_iteration = iteration_limit; // The maximum number of iterations initialized to the iteration limit
    Model so_far_the_best_model; // Current best model
    ModelScore so_far_the_best_score; // The score of the current best model
    std::unique_ptr<size_t[]> minimal_sample(new size_t[sample_size]); // The sample used for the estimation

    std::vector<size_t> pool(points_.rows);
    for (size_t point_idx = 0; point_idx < point_number; ++point_idx) {
        pool[point_idx] = point_idx;
    }

    if (points_.rows < sample_size) {
        LOG(WARNING) << "There are not enough points for applying robust estimation";
        return false;
    }

    constexpr size_t max_unsuccessful_model_generations = 50;

    // Main RANSAC iteration
    size_t iteration = 0;  // Current number of iterations
    while (minimum_iteration_number > iteration || iteration < max_iteration) {
        ++iteration;  // Increase the current iteration number

        // Sample a minimal subset
        std::vector<Model> models; // The set of estimated models
        size_t unsuccessful_model_generations = 0; // The number of unsuccessful model generations
        // Try to select a minimal sample and estimate the implied model parameters
        while (++unsuccessful_model_generations < max_unsuccessful_model_generations) {
            // Get a minimal sample randomly
            // pool: The index pool from which the minimal sample can be selected;
            // minimal_sample.get(): The minimal sample
            // sample_size: The size of a minimal sample
            if (!sampler_.sample(pool, minimal_sample.get(), sample_size)) {
                continue;
            }

            // Estimate the model from the minimal sample
            // points_: All data points
            // minimal_sample.get(): The selected minimal sample
            // &models: The estimated models
            if (estimator_.estimateModel(points_, minimal_sample.get(), &models)) { // The estimated models
                break;
            }
        }

        // If the method was not able to generate any usable models, break the cycle.
        iteration += (unsuccessful_model_generations - 1);

        // Select the so-far-the-best from the estimated models
        for (const auto &model: models) {
            ModelScore score; // The score of the current model
            Model refined_model; // The refined model parameters

            // Apply sigma-consensus to refine the model parameters by marginalizing over the noise level sigma
            bool success;
            if (RANSAC_version == "RANSAC_ORIGINAL") {
                success = sigmaConsensus(points_, model, refined_model, score, estimator_, so_far_the_best_score);
            } else {
                success = sigmaConsensusPlusPlus(points_, model, refined_model, score, estimator_, so_far_the_best_score);
            }

            // Continue if the model was rejected
            if (!success || score.score == -1) {
                continue;
            }

            // Save the iteration number when the current model is found
            score.iteration = iteration;

            // Update the best model parameters if needed
            if (so_far_the_best_score < score) {
                so_far_the_best_model = refined_model; // Update the best model parameters
                so_far_the_best_score = score; // Update the best model's score
                max_iteration = MIN(max_iteration, last_iteration_number); // Update the max iteration number, but do not allow to increase
            }
        }
    }

    obtained_model_ = so_far_the_best_model;
    iteration_number_ = iteration;
    model_score_ = so_far_the_best_score;

    return so_far_the_best_score.score > 0;
}


