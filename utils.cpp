//
// Created by yjunj on 11/8/21.
//

#include "utils.h"

cv::Mat readCorrespondences(const std::string &path_) {
    cv::Mat points_;
    std::ifstream file(path_);

    constexpr size_t dimension_number = 4 / 2;
    double x1, y1, x2, y2, a, s;
    std::string str;

    std::vector<cv::Point2d> pts1;
    std::vector<cv::Point2d> pts2;

    size_t point_number = 0;
    file >> point_number;

    for (size_t i = 0; i < point_number; ++i) {
        for (size_t dim = 0; dim < dimension_number; ++dim) {
            if (dim == 0) {
                file >> x1;
            } else if (dim == 1) {
                file >> y1;
            }
        }

        for (size_t dim = 0; dim < dimension_number; ++dim) {
            if (dim == 0) {
                file >> x2;
            } else if (dim == 1) {
                file >> y2;
            }
        }

        pts1.emplace_back(cv::Point2d(x1, y1));
        pts2.emplace_back(cv::Point2d(x2, y2));
    }

    file.close();

    points_.create(static_cast<int>(pts1.size()), 4, CV_64F);
    for (int i = 0; i < pts1.size(); ++i) {
        points_.at<double>(i, 0) = pts1[i].x;
        points_.at<double>(i, 1) = pts1[i].y;
        points_.at<double>(i, 2) = pts2[i].x;
        points_.at<double>(i, 3) = pts2[i].y;
    }
    return points_;
}


Eigen::Matrix3d loadMatrix3d(const std::string &path_) {
    Eigen::Matrix3d matrix_;
    std::ifstream infile(path_);

    size_t row = 0, column = 0;
    double element;
    while (infile >> element) {
        matrix_(row, column) = element;
        ++column;
        if (column >= 3) {
            column = 0;
            ++row;
        }
    }

    infile.close();
    return matrix_;
}


cv::Mat normalizeCorrespondences(const cv::Mat &points_, const Eigen::Matrix3d &intrinsics_src_, const Eigen::Matrix3d &intrinsics_dst_) {
    cv::Mat normalized_points_(points_.size(), CV_64F);
    const double *points_ptr = reinterpret_cast<double *>(points_.data);
    auto *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
    const Eigen::Matrix3d inverse_intrinsics_src = intrinsics_src_.inverse(),
            inverse_intrinsics_dst = intrinsics_dst_.inverse();

    double x0, y0, x1, y1;
    for (auto r = 0; r < points_.rows; ++r) {
        Eigen::Vector3d point_src, point_dst, normalized_point_src, normalized_point_dst;

        x0 = *(points_ptr++);
        y0 = *(points_ptr++);
        x1 = *(points_ptr++);
        y1 = *(points_ptr++);

        point_src << x0, y0, 1.0; // Homogeneous point in the first image
        point_dst << x1, y1, 1.0; // Homogeneous point in the second image

        // Normalized homogeneous point in the first image
        normalized_point_src = inverse_intrinsics_src * point_src;
        // Normalized homogeneous point in the second image
        normalized_point_dst = inverse_intrinsics_dst * point_dst;

        // The second four columns contain the normalized coordinates.
        *(normalized_points_ptr++) = normalized_point_src(0);
        *(normalized_points_ptr++) = normalized_point_src(1);
        *(normalized_points_ptr++) = normalized_point_dst(0);
        *(normalized_points_ptr++) = normalized_point_dst(1);
    }
    return normalized_points_;
}


void drawMatches(const cv::Mat &points_, const std::vector<int> &labeling_,
                 const cv::Mat &image1_, const cv::Mat &image2_, cv::Mat &out_image_) {
    const size_t N = points_.rows;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;

    keypoints1.reserve(N);
    keypoints2.reserve(N);
    matches.reserve(N);

    for (auto pt_idx = 0; pt_idx < N; ++pt_idx) {
        // Collect the points which has label 1 (i.e. inlier)
        if (!labeling_[pt_idx]) {
            continue;
        }

        const double x1 = points_.at<double>(pt_idx, 0);
        const double y1 = points_.at<double>(pt_idx, 1);
        const double x2 = points_.at<double>(pt_idx, 2);
        const double y2 = points_.at<double>(pt_idx, 3);
        const size_t n = keypoints1.size();

        keypoints1.emplace_back(cv::KeyPoint(cv::Point_<double>(x1, y1), 0));
        keypoints2.emplace_back(cv::KeyPoint(cv::Point_<double>(x2, y2), 0));
        matches.emplace_back(cv::DMatch(static_cast<int>(n), static_cast<int>(n), 0));
    }

    // Draw the matches using OpenCV's built-in function
    cv::drawMatches(image1_, keypoints1, image2_, keypoints2, matches, out_image_);
}

void showImage(const cv::Mat &image_,
               std::string window_name_,
               int max_width_,
               int max_height_,
               bool wait_) {
    // Resizing the window to fit into the screen if needed
    int window_width = image_.cols,
            window_height = image_.rows;
    if (static_cast<double>(image_.cols) / max_width_ > 1.0 &&
        static_cast<double>(image_.cols) / max_width_ >
        static_cast<double>(image_.rows) / max_height_) {
        window_width = max_width_;
        window_height = static_cast<int>(window_width * static_cast<double>(image_.rows) / static_cast<double>(image_.cols));
    } else if (static_cast<double>(image_.rows) / max_height_ > 1.0 &&
               static_cast<double>(image_.cols) / max_width_ <
               static_cast<double>(image_.rows) / max_height_) {
        window_height = max_height_;
        window_width = static_cast<int>(window_height * static_cast<double>(image_.cols) / static_cast<double>(image_.rows));
    }

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_, window_width, window_height);
    cv::imshow(window_name_, image_);
    if (wait_) {
        cv::waitKey(0);
    }
}