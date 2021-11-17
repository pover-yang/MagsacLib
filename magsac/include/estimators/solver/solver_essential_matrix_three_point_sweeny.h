//
// Created by yjunj on 10/18/21.
//
#include <vector>
#include <Eigen/Dense>

#include "solver_engine.h"


namespace estimator::solver {
    class EssentialMatrixThreePointSweenySolver : public SolverEngine {
    public:
        EssentialMatrixThreePointSweenySolver() = default;

        ~EssentialMatrixThreePointSweenySolver() = default;

        static constexpr size_t sampleSize() {
            return 3;
        }

        static constexpr size_t maximumSolutions() {
            return 6;
        }

        static constexpr bool returnMultipleModels() {
            return maximumSolutions() > 1;
        }

        static bool estimateModel(const cv::Mat &data,
                                  const size_t *sample,
                                  size_t sample_number,
                                  std::vector<Model> &models,
                                  const double *weights_ = nullptr);

    protected:
        static void essential_from_motion(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, Eigen::Matrix3d *E);

        static int qep_div_1_q2(const Eigen::Matrix3d &A,
                                const Eigen::Matrix3d &B,
                                const Eigen::Matrix3d &C,
                                double eig_vals[4],
                                Eigen::Matrix<double, 3, 4> *eig_vecs);

        static int solve_quartic_real(double b, double c, double d, double e, double roots[4]);

        static void solve_cubic_single_real(double c2, double c1, double c0, double &root);

        static void detpoly3(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B, double coeffs[7]);

        static double sign(double z);
    };

    inline bool EssentialMatrixThreePointSweenySolver::estimateModel(
            const cv::Mat &data,
            const size_t *sample,
            size_t sample_number,
            std::vector<Model> &models,
            const double *weights_) {

        const Eigen::Vector3d axis(0, 0, 1);
        const double *data_ptr = reinterpret_cast<double *>(data.data);
        const int cols = data.cols;
        Eigen::Matrix<double, 3, 3> M, C, K;

        int offset;
        double x1, y1, x2, y2;
        for (int i = 0; i < sampleSize(); ++i) {
            offset = sample[i] * cols;
            x1 = data_ptr[offset];
            y1 = data_ptr[offset + 1];
            x2 = data_ptr[offset + 2];
            y2 = data_ptr[offset + 3];
            const Eigen::Vector3d p1(x1, y1, 1);
            const Eigen::Vector3d p2(x2, y2, 1);

            M.row(i) = p2.cross(p1);
            C.row(i) = 2.0 * p2.cross(axis.cross(p1));
            K.row(i) = 2.0 * p1.dot(axis) * p2.cross(axis) - p2.cross(p1);
        }
        double eig_vals[4];
        Eigen::Matrix<double, 3, 4> eig_vecs;

        std::vector<double> eigenvalues;
        std::vector<Eigen::Vector3d> eigenvectors;

        const int n_roots = qep_div_1_q2(M, C, K, eig_vals, &eig_vecs);

        models.clear();
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        Eigen::Matrix3d essential_mtx;
        for (int i = 0; i < n_roots; ++i) {
            const double q = eig_vals[i];
            const double q2 = q * q;
            const double inv_norm = 1.0 / (1 + q2);
            const double cq = (q2 - 1) * inv_norm;
            const double sq = 2 * q * inv_norm;

            R.setIdentity();
            R(0, 0) = cq;
            R(0, 1) = -sq;
            R(1, 0) = sq;
            R(1, 1) = cq;
            t = eig_vecs.col(i);
            essential_from_motion(R, t, &essential_mtx);
            EssentialMatrix model;
            model.descriptor = essential_mtx;
            models.push_back(model);
        }
        return !models.empty();
    };

    inline void
    EssentialMatrixThreePointSweenySolver::essential_from_motion(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, Eigen::Matrix3d *E) {
        *E <<
           0.0, -t(2), t(1),
                t(2), 0.0, -t(0),
                t(1), t(0), 0.0;
        *E = (*E) * R;
    }

    inline int EssentialMatrixThreePointSweenySolver::qep_div_1_q2(const Eigen::Matrix<double, 3, 3> &A,
                                                                   const Eigen::Matrix<double, 3, 3> &B,
                                                                   const Eigen::Matrix<double, 3, 3> &C,
                                                                   double eig_vals[4],
                                                                   Eigen::Matrix<double, 3, 4> *eig_vecs) {

        double coeffs[7];

        Eigen::Matrix<double, 3, 3> Ainv = A.inverse();
        detpoly3(Ainv * B, Ainv * C, coeffs);

        int n_roots = solve_quartic_real(coeffs[5], coeffs[2] - coeffs[0], coeffs[1], coeffs[0],
                                         eig_vals);

        Eigen::Matrix<double, 3, 3> M;
        bool invertible;
        for (int i = 0; i < n_roots; ++i) {
            M = (eig_vals[i] * eig_vals[i]) * A + eig_vals[i] * B + C;

            Eigen::Vector3d t = M.row(0).cross(M.row(1)).normalized();
            if (std::abs(M.row(2) * t) > 1e-8) {
                t = M.row(0).cross(M.row(2)).normalized();
                if (std::abs(M.row(1) * t) > 1e-8) {
                    t = M.row(1).cross(M.row(2)).normalized();
                }
            }

            eig_vecs->col(i) = t;
        }

        return n_roots;
    }

    inline int EssentialMatrixThreePointSweenySolver::solve_quartic_real(double b, double c, double d, double e, double roots[4]) {

        // Find depressed quartic
        double p = c - 3.0 * b * b / 8.0;
        double q = b * b * b / 8.0 - 0.5 * b * c + d;
        double r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;

        // Resolvent cubic is now
        // U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
        double bb = 2.0 * p;
        double cc = p * p - 4.0 * r;
        double dd = -q * q;

        // Solve resolvent cubic
        double u2;
        solve_cubic_single_real(bb, cc, dd, u2);

        if (u2 < 0) {
            return 0;
        }

        double u = sqrt(u2);

        double s = -u;
        double t = (p + u * u + q / u) / 2.0;
        double v = (p + u * u - q / u) / 2.0;

        int sols = 0;
        double disc = u * u - 4.0 * v;
        if (disc > 0) {
            roots[0] = (-u - sign(u) * std::sqrt(disc)) / 2.0;
            roots[1] = v / roots[0];
            sols += 2;
        }
        disc = s * s - 4.0 * t;
        if (disc > 0) {
            roots[sols] = (-s - sign(s) * std::sqrt(disc)) / 2.0;
            roots[sols + 1] = t / roots[sols];
            sols += 2;
        }

        for (int i = 0; i < sols; i++) {
            roots[i] = roots[i] - b / 4.0;

            // do one step of newton refinement
            double x = roots[i];
            double x2 = x * x;
            double x3 = x * x2;
            double dx = -(x2 * x2 + b * x3 + c * x2 + d * x + e) / (4.0 * x3 + 3.0 * b * x2 + 2.0 * c * x + d);
            roots[i] = x + dx;
        }
        return sols;
    }

    inline void EssentialMatrixThreePointSweenySolver::solve_cubic_single_real(double c2, double c1, double c0, double &root) {
        double a = c1 - c2 * c2 / 3.0;
        double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
        double c = b * b / 4.0 + a * a * a / 27.0;
        if (c > 0) {
            c = std::sqrt(c);
            b *= -0.5;
            root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
        } else {
            c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
            root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        }
    }

    inline void EssentialMatrixThreePointSweenySolver::detpoly3(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B,
                                                                double coeffs[7]) {
        coeffs[0] = B(0, 0) * B(1, 1) * B(2, 2) - B(0, 0) * B(1, 2) * B(2, 1) - B(0, 1) * B(1, 0) * B(2, 2) +
                    B(0, 1) * B(1, 2) * B(2, 0) + B(0, 2) * B(1, 0) * B(2, 1) - B(0, 2) * B(1, 1) * B(2, 0);
        coeffs[1] = A(0, 0) * B(1, 1) * B(2, 2) - A(0, 0) * B(1, 2) * B(2, 1) - A(0, 1) * B(1, 0) * B(2, 2) +
                    A(0, 1) * B(1, 2) * B(2, 0) + A(0, 2) * B(1, 0) * B(2, 1) - A(0, 2) * B(1, 1) * B(2, 0) -
                    A(1, 0) * B(0, 1) * B(2, 2) + A(1, 0) * B(0, 2) * B(2, 1) + A(1, 1) * B(0, 0) * B(2, 2) -
                    A(1, 1) * B(0, 2) * B(2, 0) - A(1, 2) * B(0, 0) * B(2, 1) + A(1, 2) * B(0, 1) * B(2, 0) +
                    A(2, 0) * B(0, 1) * B(1, 2) - A(2, 0) * B(0, 2) * B(1, 1) - A(2, 1) * B(0, 0) * B(1, 2) +
                    A(2, 1) * B(0, 2) * B(1, 0) + A(2, 2) * B(0, 0) * B(1, 1) - A(2, 2) * B(0, 1) * B(1, 0);
        coeffs[2] =
                B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0) + B(0, 0) * B(2, 2) - B(0, 2) * B(2, 0) + B(1, 1) * B(2, 2) -
                B(1, 2) * B(2, 1) + A(0, 0) * A(1, 1) * B(2, 2) - A(0, 0) * A(1, 2) * B(2, 1) -
                A(0, 0) * A(2, 1) * B(1, 2) + A(0, 0) * A(2, 2) * B(1, 1) - A(0, 1) * A(1, 0) * B(2, 2) +
                A(0, 1) * A(1, 2) * B(2, 0) + A(0, 1) * A(2, 0) * B(1, 2) - A(0, 1) * A(2, 2) * B(1, 0) +
                A(0, 2) * A(1, 0) * B(2, 1) - A(0, 2) * A(1, 1) * B(2, 0) - A(0, 2) * A(2, 0) * B(1, 1) +
                A(0, 2) * A(2, 1) * B(1, 0) + A(1, 0) * A(2, 1) * B(0, 2) - A(1, 0) * A(2, 2) * B(0, 1) -
                A(1, 1) * A(2, 0) * B(0, 2) + A(1, 1) * A(2, 2) * B(0, 0) + A(1, 2) * A(2, 0) * B(0, 1) -
                A(1, 2) * A(2, 1) * B(0, 0);
        coeffs[3] =
                A(0, 0) * B(1, 1) - A(0, 1) * B(1, 0) - A(1, 0) * B(0, 1) + A(1, 1) * B(0, 0) + A(0, 0) * B(2, 2) -
                A(0, 2) * B(2, 0) - A(2, 0) * B(0, 2) + A(2, 2) * B(0, 0) + A(1, 1) * B(2, 2) - A(1, 2) * B(2, 1) -
                A(2, 1) * B(1, 2) + A(2, 2) * B(1, 1) + A(0, 0) * A(1, 1) * A(2, 2) - A(0, 0) * A(1, 2) * A(2, 1) -
                A(0, 1) * A(1, 0) * A(2, 2) + A(0, 1) * A(1, 2) * A(2, 0) + A(0, 2) * A(1, 0) * A(2, 1) -
                A(0, 2) * A(1, 1) * A(2, 0);
        coeffs[4] = B(0, 0) + B(1, 1) + B(2, 2) + A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) -
                    A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
        coeffs[5] = A(0, 0) + A(1, 1) + A(2, 2);
        coeffs[6] = 1.0;
    }

    inline double EssentialMatrixThreePointSweenySolver::sign(const double z) {
        return z < 0 ? -1.0 : 1.0;
    }
}
