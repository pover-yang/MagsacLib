// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Eigen>

#include "types.h"
#include "statistics.h"

namespace gcransac {
    namespace utils {
        /*
            Function declaration
        */

        template<typename T, size_t N, size_t M>
        bool loadMatrix(const std::string &path_,
                        Eigen::Matrix<T, N, M> &matrix_);

        void normalizeCorrespondences(const cv::Mat &points_,
                                      const Eigen::Matrix3d &intrinsics_src_,
                                      const Eigen::Matrix3d &intrinsics_dst_,
                                      cv::Mat &normalized_points_);

        template<typename T, size_t N, size_t M>
        bool loadMatrix(const std::string &path_,
                        Eigen::Matrix<T, N, M> &matrix_) {
            std::ifstream infile(path_);

            if (!infile.is_open())
                return false;

            size_t row = 0,
                    column = 0;
            double element;

            while (infile >> element) {
                matrix_(row, column) = element;
                ++column;
                if (column >= M) {
                    column = 0;
                    ++row;
                }
            }

            infile.close();

            return row == N &&
                   column == 0;
        }

        void normalizeCorrespondences(const cv::Mat &points_,
                                      const Eigen::Matrix3d &intrinsics_src_,
                                      const Eigen::Matrix3d &intrinsics_dst_,
                                      cv::Mat &normalized_points_) {
            const double *points_ptr = reinterpret_cast<double *>(points_.data);
            double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
            const Eigen::Matrix3d inverse_intrinsics_src = intrinsics_src_.inverse(),
                    inverse_intrinsics_dst = intrinsics_dst_.inverse();

            // Most likely, this is not the fastest solution, but it does
            // not affect the speed of Graph-cut RANSAC, so not a crucial part of
            // this example.
            double x0, y0, x1, y1;
            for (auto r = 0; r < points_.rows; ++r) {
                Eigen::Vector3d point_src,
                        point_dst,
                        normalized_point_src,
                        normalized_point_dst;

                x0 = *(points_ptr++);
                y0 = *(points_ptr++);
                x1 = *(points_ptr++);
                y1 = *(points_ptr++);

                point_src << x0, y0, 1.0; // Homogeneous point in the first image
                point_dst << x1, y1, 1.0; // Homogeneous point in the second image

                // Normalized homogeneous point in the first image
                normalized_point_src =
                        inverse_intrinsics_src * point_src;
                // Normalized homogeneous point in the second image
                normalized_point_dst =
                        inverse_intrinsics_dst * point_dst;

                // The second four columns contain the normalized coordinates.
                *(normalized_points_ptr++) = normalized_point_src(0);
                *(normalized_points_ptr++) = normalized_point_src(1);
                *(normalized_points_ptr++) = normalized_point_dst(0);
                *(normalized_points_ptr++) = normalized_point_dst(1);
            }
        }
    }
}
