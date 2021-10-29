#include <string.h>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "magsac_utils.h"
#include "utils.h"
#include "magsac.h"

#include "samplers/progressive_napsac_sampler.h"
#include "samplers/uniform_sampler.h"
#include "neighborhood/flann_neighborhood_graph.h"
#include "model.h"
#include "estimators.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

// A method applying MAGSAC for essential matrix estimation to one of the built-in scenes
void testEssentialMatrixFitting(
	double ransac_confidence_,
	double maximum_threshold_,
	bool use_magsac_plus_plus_ = true,
	double drawing_threshold_ = 2);

// A method applying OpenCV for essential matrix estimation to one of the built-in scenes
void opencvEssentialMatrixFitting(
	double ransac_confidence_,
	double threshold_);

// Running tests
void runTest(double ransac_confidence_,
	double drawing_threshold_);

int main(int argc, char** argv)
{	
	// Parsing the flags & Initialize Google's logging library.
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	const double ransac_confidence = 0.90; // The required confidence in the results
	const double drawing_threshold = 3.00; // Threshold for visualization which not used by the algorithm

    runTest(ransac_confidence, drawing_threshold);
	return 0;
} 

void runTest(const double ransac_confidence_, // The confidence required in the results
	const double drawing_threshold_) // The threshold used for selecting the inliers when they are drawn
{
    // Apply the homography estimation method built into OpenCV
    LOG(INFO) << "1. Running OpenCV's RANSAC with threshold " << drawing_threshold_ << " px";
    opencvEssentialMatrixFitting(ransac_confidence_, drawing_threshold_); // The maximum sigma value allowed in MAGSAC

    // Apply MAGSAC with a reasonably set maximum threshold
    LOG(INFO) << "2. Running MAGSAC with fairly high maximum threshold (" << 5 << " px)";
    testEssentialMatrixFitting(ransac_confidence_, // The required confidence in the results
        2.0, // The maximum sigma value allowed in MAGSAC
        false, // MAGSAC should be used
        drawing_threshold_); // The inlier threshold for visualization.

    // Apply MAGSAC with a reasonably set maximum threshold
    LOG(INFO) << "3. Running MAGSAC++ with fairly high maximum threshold (" << 5 << " px)";
    testEssentialMatrixFitting(ransac_confidence_, // The required confidence in the results
        2.0, // The maximum sigma value allowed in MAGSAC
        true, // MAGSAC++ should be used
        drawing_threshold_); // The inlier threshold for visualization.
    cv::waitKey(0);

}

// A method applying MAGSAC for essential matrix estimation to one of the built-in scenes
void testEssentialMatrixFitting(
	double ransac_confidence_,
	double maximum_threshold_,
	bool use_magsac_plus_plus_, // A flag to decide if MAGSAC++ or MAGSAC should be used
	double drawing_threshold_)
{
	// Load the images of the current test scene
	cv::Mat image1 = cv::imread("../../data/essential_matrix/fountain1.jpg");
	cv::Mat image2 = cv::imread("../../data/essential_matrix/fountain2.jpg");

	cv::Mat points; // The point correspondences, each is of format x1 y1 x2 y2
	Eigen::Matrix3d intrinsics_source, // The intrinsic parameters of the source camera
		intrinsics_destination; // The intrinsic parameters of the destination camera

	// A function loading the points from files
	readPoints<4>("../../data/essential_matrix/fountain_pts.txt", points);

	// Loading the intrinsic camera matrices
	static const std::string source_intrinsics_path = "../../data/essential_matrix/fountain1.K";
	if (!gcransac::utils::loadMatrix<double, 3, 3>(source_intrinsics_path,
		intrinsics_source))
	{
		LOG(ERROR) << "An error occured when loading the intrinsics camera matrix from " << source_intrinsics_path << ".";
		return;
	}

	static const std::string destination_intrinsics_path = "../../data/essential_matrix/fountain2.K";
	if (!gcransac::utils::loadMatrix<double, 3, 3>(destination_intrinsics_path,
		intrinsics_destination))
	{
		LOG(ERROR) << "An error occured when loading the intrinsics camera matrix from " << destination_intrinsics_path << ".";
		return;
	}

	// Normalize the point coordinates by the intrinsic matrices
	cv::Mat normalized_points(points.size(), CV_64F);
	gcransac::utils::normalizeCorrespondences(points,
		intrinsics_source,
		intrinsics_destination,
		normalized_points);

	// Normalize the threshold by the average of the focal lengths
	const double normalizing_multiplier = 1.0 / ((intrinsics_source(0, 0) + intrinsics_source(1, 1) +
		intrinsics_destination(0, 0) + intrinsics_destination(1, 1)) / 4.0);
	const double normalized_maximum_threshold =
		maximum_threshold_ * normalizing_multiplier;
	const double normalized_drawing_threshold =
		drawing_threshold_ * normalizing_multiplier;

	// The number of points in the datasets
	const size_t N = points.rows; // The number of points in the scene

	// The robust homography estimator class containing the function for the fitting and residual calculation
	magsac::utils::DefaultEssentialMatrixEstimator estimator(
		intrinsics_source,
		intrinsics_destination,
		0.0); 
	gcransac::EssentialMatrix model; // The estimated model
	
	LOG(INFO) << "Estimated model = " << "essential matrix";

	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		{ static_cast<double>(image1.cols), // The width of the source image
			static_cast<double>(image1.rows), // The height of the source image
			static_cast<double>(image2.cols), // The width of the destination image
			static_cast<double>(image2.rows) },  // The height of the destination image
		0.5); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

	MAGSAC<cv::Mat, magsac::utils::DefaultEssentialMatrixEstimator> magsac
		(use_magsac_plus_plus_ ?
			MAGSAC<cv::Mat, magsac::utils::DefaultEssentialMatrixEstimator>::MAGSAC_PLUS_PLUS :
			MAGSAC<cv::Mat, magsac::utils::DefaultEssentialMatrixEstimator>::MAGSAC_ORIGINAL);
	magsac.setMaximumThreshold(normalized_maximum_threshold); // The maximum noise scale sigma allowed
	magsac.setReferenceThreshold(magsac.getReferenceThreshold() * normalizing_multiplier); // The reference threshold inside MAGSAC++ should also be normalized.
	magsac.setIterationLimit(1e4); // Iteration limit to interrupt the cases when the algorithm run too long.

	int iteration_number = 0; // Number of iterations required
	ModelScore score; // The model score

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	magsac.run(normalized_points, // The data points
		ransac_confidence_, // The required confidence in the results
		estimator, // The used estimator
		main_sampler, // The sampler used for selecting minimal samples in each iteration
		model, // The estimated model
		iteration_number, // The number of iterations
		score); // The score of the estimated model
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	LOG(INFO) << "Actual number of iterations drawn by MAGSAC at " << ransac_confidence_ << " confidence = " << iteration_number;
	LOG(INFO) << "Elapsed time = " << elapsed_seconds.count() << " seconds";
	
	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	std::vector<int> obtained_labeling(points.rows, 0);
	size_t inlier_number = 0;

	for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx)
	{
		// Computing the residual of the point given the estimated model
		auto residual = estimator.residual(normalized_points.row(pt_idx),
			model.descriptor);

		// Change the label to 'inlier' if the residual is smaller than the threshold
		if (normalized_drawing_threshold >= residual)
		{
			obtained_labeling[pt_idx] = 1;
			++inlier_number;
		}
	}

	LOG(INFO) << "Number of points closer than " << drawing_threshold_ << " is " << static_cast<int>(inlier_number);

    // Draw the matches to the images
    cv::Mat out_image;
    drawMatches<double, int>(points, obtained_labeling, image1, image2, out_image);

    // Show the matches
    std::string window_name = "Visualization with threshold = " + std::to_string(drawing_threshold_) + " px; Maximum threshold is = " + std::to_string(maximum_threshold_);
    showImage(out_image,
        window_name,
        1600,
        900);
    out_image.release();

	// Clean up the memory occupied by the images
	image1.release();
	image2.release();
}


// A method applying OpenCV for essential matrix estimation to one of the built-in scenes
void opencvEssentialMatrixFitting(
	double ransac_confidence_,
	double threshold_)
{
	// Load the images of the current test scene
	cv::Mat image1 = cv::imread("../../data/essential_matrix/fountain1.jpg");
	cv::Mat image2 = cv::imread("../../data/essential_matrix/fountain2.jpg");

	cv::Mat points; // The point correspondences, each is of format x1 y1 x2 y2
	Eigen::Matrix3d intrinsics_source, // The intrinsic parameters of the source camera
		intrinsics_destination; // The intrinsic parameters of the destination camera
	
	// A function loading the points from files
	readPoints<4>("../../data/essential_matrix/fountain_pts.txt", points);

	// Loading the intrinsic camera matrices
	static const std::string source_intrinsics_path = "../../data/essential_matrix/fountain1.K";
	if (!gcransac::utils::loadMatrix<double, 3, 3>(source_intrinsics_path,
		intrinsics_source))
	{
		LOG(ERROR) << "An error occured when loading the intrinsics camera matrix from " << source_intrinsics_path << ".";
		return;
	}

	static const std::string destination_intrinsics_path = "../../data/essential_matrix/fountain2.K";
	if (!gcransac::utils::loadMatrix<double, 3, 3>(destination_intrinsics_path,
		intrinsics_destination))
	{
		LOG(ERROR) << "An error occured when loading the intrinsics camera matrix from " << destination_intrinsics_path << ".";
		return;
	}

	// Normalize the point coordinates by the intrinsic matrices
	cv::Mat normalized_points(points.size(), CV_64F);
	gcransac::utils::normalizeCorrespondences(points,
		intrinsics_source,
		intrinsics_destination,
		normalized_points);

	cv::Mat cv_intrinsics_source(3, 3, CV_64F);
	cv_intrinsics_source.at<double>(0, 0) = intrinsics_source(0, 0);
	cv_intrinsics_source.at<double>(0, 1) = intrinsics_source(0, 1);
	cv_intrinsics_source.at<double>(0, 2) = intrinsics_source(0, 2);
	cv_intrinsics_source.at<double>(1, 0) = intrinsics_source(1, 0);
	cv_intrinsics_source.at<double>(1, 1) = intrinsics_source(1, 1);
	cv_intrinsics_source.at<double>(1, 2) = intrinsics_source(1, 2);
	cv_intrinsics_source.at<double>(2, 0) = intrinsics_source(2, 0);
	cv_intrinsics_source.at<double>(2, 1) = intrinsics_source(2, 1);
	cv_intrinsics_source.at<double>(2, 2) = intrinsics_source(2, 2);

	const size_t N = points.rows;

	const double normalized_threshold =
		threshold_ / ((intrinsics_source(0, 0) + intrinsics_source(1, 1) +
			intrinsics_destination(0, 0) + intrinsics_destination(1, 1)) / 4.0);

	// Define location of sub matrices in data matrix
	cv::Rect roi1(0, 0, 2, N);
	cv::Rect roi2(2, 0, 2, N);

	std::vector<uchar> obtained_labeling(points.rows, 0);
	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();

	// Estimating the homography matrix by OpenCV's RANSAC
	cv::Mat cv_essential_matrix = cv::findEssentialMat(cv::Mat(normalized_points, roi1), // The points in the first image
		cv::Mat(normalized_points, roi2), // The points in the second image
		cv::Mat::eye(3, 3, CV_64F), // The intrinsic camera matrix of the source image
        cv::RANSAC, // The method used for the fitting
		ransac_confidence_, // The RANSAC confidence
		normalized_threshold, // The inlier-outlier threshold
		obtained_labeling); // The obtained labeling

	// Convert cv::Mat to Eigen::Matrix3d 
	Eigen::Matrix3d essential_matrix =
		Eigen::Map<Eigen::Matrix3d>(cv_essential_matrix.ptr<double>(), 3, 3);

	end = std::chrono::system_clock::now();

	// Calculate the processing time of OpenCV
	std::chrono::duration<double> elapsed_seconds = end - start;
	
	LOG(INFO) << "Elapsed time = " << elapsed_seconds.count() << " seconds";

	size_t inlier_number = 0;
	
	// Visualization part.
	for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx)
	{
		// Change the label to 'inlier' if the residual is smaller than the threshold
		if (obtained_labeling[pt_idx])
			++inlier_number;
	}

	LOG(INFO) << "Number of points closer than " << threshold_ << " is " << static_cast<int>(inlier_number);

    // Draw the matches to the images
    cv::Mat out_image;
    drawMatches<double, uchar>(points, obtained_labeling, image1, image2, out_image);

    // Show the matches
    std::string window_name = "Threshold = " + std::to_string(threshold_) + " px";
    showImage(out_image, window_name, 1600, 900);
    out_image.release();

	// Clean up the memory occupied by the images
	image1.release();
	image2.release();
}
