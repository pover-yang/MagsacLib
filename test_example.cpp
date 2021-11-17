#include <random>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "model.h"
#include "utils.h"
#include "magsac.h"
#include "estimators/essential_estimator.h"
#include "samplers/uniform_sampler.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

// A method applying MAGSAC for essential matrix estimation to one of the built-in scenes
void testEssentialMatrixFitting(double ransac_confidence_, double maximum_threshold_, bool use_magsac_plus_plus_,
                                bool use_real_data);

cv::Mat generateTestData(Eigen::Matrix3d &intrinsics_src, Eigen::Matrix3d &intrinsics_dst,
                         bool use_real_data, cv::Mat &img1, cv::Mat &img2, cv::Mat &points);


int main(int argc, char **argv) {
    // Parsing the flags & Initialize Google's logging library.
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    const double ransac_confidence = 0.90; // The required confidence in the results
    const double maximum_threshold = 5; // The maximum sigma value allowed in MAGSAC

    // Apply MAGSAC with a reasonably set maximum threshold
    LOG(INFO) << "Running MAGSAC with Three Point Solver";
    testEssentialMatrixFitting(ransac_confidence, maximum_threshold, false, false);
    cv::waitKey(0);

//    // Apply MAGSAC with a reasonably set maximum threshold
//    LOG(INFO) << "Running MAGSAC with Five Point Solver";
//    testEssentialMatrixFitting(ransac_confidence, 2.0, false, true);

    return 0;
}

// A method applying MAGSAC for essential matrix estimation to one of the built-in scenes
void testEssentialMatrixFitting(
        double ransac_confidence_,  // The required confidence in the results
        double maximum_threshold_,  // The maximum sigma value allowed in MAGSAC
        bool use_magsac_plus_plus_, // A flag to decide if MAGSAC++ or MAGSAC should be used
        bool use_real_data) {
    cv::Mat norm_points, points, img1, img2;
    Eigen::Matrix3d intrinsics_src, intrinsics_dst;
    norm_points = generateTestData(intrinsics_src, intrinsics_dst, use_real_data, img1, img2, points);

    // Initialize the sampler used for selecting minimal samples
    sampler::UniformSampler main_sampler(&norm_points);

    // The robust essential matrix estimator class containing the function for the fitting and residual calculation
//    estimator::DefaultEssentialMatrixEstimator estimator(intrinsics_src, intrinsics_dst, 0.0);
//    MAGSAC<estimator::DefaultEssentialMatrixEstimator> magsac(use_magsac_plus_plus_ ? "MAGSAC_PLUS_PLUS" : "MAGSAC_ORIGINAL");

    estimator::TestEssentialMatrixEstimator estimator(intrinsics_src, intrinsics_dst, 0.0);
    MAGSAC<estimator::TestEssentialMatrixEstimator> magsac(use_magsac_plus_plus_ ? "MAGSAC_PLUS_PLUS" : "MAGSAC_ORIGINAL");

    // Normalize the threshold by the average of the focal lengths
    const double normalizing_multiplier = 1.0 / ((intrinsics_src(0, 0) + intrinsics_src(1, 1) +
                                                  intrinsics_dst(0, 0) + intrinsics_dst(1, 1)) / 4.0);
    magsac.setReferenceThreshold(magsac.getReferenceThreshold() * normalizing_multiplier);
    magsac.setMaximumThreshold(maximum_threshold_ * normalizing_multiplier); // The maximum noise scale sigma allowed
    magsac.setIterationLimit(1e4); // Iteration limit to interrupt the cases when the algorithm run too long.

    int iteration_number = 0; // Number of iterations required
    EssentialMatrix model; // The estimated model
    ModelScore score; // The model score

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    magsac.run(norm_points, // The data points
               ransac_confidence_, // The required confidence in the results
               estimator, // The used estimator
               main_sampler, // The sampler used for selecting minimal samples in each iteration
               model, // The estimated model parameters
               iteration_number, // The number of iterations done
               score); // The score of the estimated model
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    LOG(INFO) << "Actual number of iterations drawn by MAGSAC at " << ransac_confidence_ << " confidence = " << iteration_number;
    LOG(INFO) << "Elapsed time = " << elapsed_seconds.count() << " seconds";

    // Visualization part.
    const double drawing_threshold_ = 2; // Threshold for visualization which not used by the algorithm
    const double normalized_drawing_threshold = drawing_threshold_ * normalizing_multiplier;

    std::vector<int> obtained_labeling(norm_points.rows, 0);
    size_t inlier_number = 0;

    for (auto pt_idx = 0; pt_idx < norm_points.rows; ++pt_idx) {
        // Computing the residual of the point given the estimated model
        auto residual = estimator.residual(norm_points.row(pt_idx), model.descriptor);

        // Change the label to 'inlier' if the residual is smaller than the threshold
        if (normalized_drawing_threshold >= residual) {
            obtained_labeling[pt_idx] = 1;
            ++inlier_number;
        }
    }
    LOG(INFO) << "Number of points closer than " << drawing_threshold_ << " is " << static_cast<int>(inlier_number);

    if (use_real_data) {
        // Draw the matches to the images
        cv::Mat out_image;
        drawMatches(points, obtained_labeling, img1, img2, out_image);

        // Show the matches
        std::string window_name =
                "Threshold = " + std::to_string(drawing_threshold_) + " px; Maximum threshold is = " + std::to_string(maximum_threshold_);
        showImage(out_image, window_name, 1600, 900);
        out_image.release();

        // Clean up the memory occupied by the images
        img1.release();
        img2.release();
    }
}


cv::Mat generateTestData(Eigen::Matrix3d &intrinsics_src, Eigen::Matrix3d &intrinsics_dst,
                         bool use_real_data, cv::Mat &img1, cv::Mat &img2, cv::Mat &points) {
    cv::Mat normalized_points;
    if (use_real_data) {
        // Load the images of the current test scene
        img1 = cv::imread("../../data/essential_matrix/fountain1.jpg");
        img2 = cv::imread("../../data/essential_matrix/fountain2.jpg");

        // Loading the points from files
        points = readCorrespondences("../../data/essential_matrix/fountain_pts.txt");

        // Loading the intrinsic camera matrices
        intrinsics_src = loadMatrix3d("../../data/essential_matrix/fountain1.K");
        intrinsics_dst = loadMatrix3d("../../data/essential_matrix/fountain2.K");

        // Normalize the point coordinates by the intrinsic matrices
        normalized_points = normalizeCorrespondences(points, intrinsics_src, intrinsics_dst);
    } else {
        intrinsics_src << 4523.2365, 0, 1920, 0, 4523.2365, 1080, 0, 0, 1;
        intrinsics_dst << 4523.2365, 0, 1920, 0, 4523.2365, 1080, 0, 0, 1;

        int num_corresponds = 500;
        Eigen::Vector3d axis(0, 0, 1);
        double rotation_angle = 45;
        Eigen::AngleAxisd relative_rotation(rotation_angle / 180 * M_PI, axis);

        double x = (double(rand()) / RAND_MAX - 0.5) * 2 * 10;
        double y = (double(rand()) / RAND_MAX - 0.5) * 2 * 10;
        double z = (double(rand()) / RAND_MAX - 0.5) * 2 * 0.5;
        Eigen::Vector3d relative_translation(x, y, z);
        double jitter_degree = 2;
        int height = 10;
        double noise_max = 2;
        normalized_points = synthesisCorresponds(num_corresponds, relative_rotation, relative_translation, intrinsics_src,
                                                 jitter_degree, height, noise_max);

    }
    return normalized_points;
}

