//
// Created by yjunj on 11/8/21.
//

#ifndef MAGSACLIB_UTILS_H
#define MAGSACLIB_UTILS_H

#include <fstream>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

cv::Mat readCorrespondences(const std::string &path_);

Eigen::Matrix3d loadMatrix3d(const std::string &path_);

cv::Mat normalizeCorrespondences(const cv::Mat &points_,
                                 const Eigen::Matrix3d &intrinsics_src_,
                                 const Eigen::Matrix3d &intrinsics_dst_);

void drawMatches(const cv::Mat &points_, const std::vector<int> &labeling_,
                 const cv::Mat &image1_, const cv::Mat &image2_, cv::Mat &out_image_);

void showImage(const cv::Mat &image_, std::string window_name_, int max_width_, int max_height_, bool wait_ = true);

#endif //MAGSACLIB_UTILS_H
