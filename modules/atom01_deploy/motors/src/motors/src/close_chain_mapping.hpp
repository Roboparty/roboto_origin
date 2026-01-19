#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <map>

using namespace std;

struct InsKinematicsResult
{
    std::vector<Eigen::Vector3d> r_A;
    std::vector<Eigen::Vector3d> r_B;
    std::vector<Eigen::Vector3d> r_C;
    std::vector<Eigen::Vector3d> r_bar;
    std::vector<Eigen::Vector3d> r_rod;
    Eigen::Vector2d THETA;
};

struct ForwardMappingResult
{
    int count;
    Eigen::Vector2d ankle_joint_ori;
    std::vector<Eigen::MatrixXd> Jac;
};

class Decouple
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Decouple() {}
    void printVector3d(const Eigen::Vector3d &vec);

    void printKinematicsResult(const InsKinematicsResult &result);

    InsKinematicsResult inverse_kinematics(double q_roll, double q_pitch, bool leftLegFlag);

    std::vector<Eigen::MatrixXd> jacobian(const std::vector<Eigen::Vector3d> &r_C, const std::vector<Eigen::Vector3d> &r_bar,
                                          const std::vector<Eigen::Vector3d> &r_rod, double q_pitch);

    std::pair<Eigen::Vector2d, std::vector<Eigen::MatrixXd>> getDecouple(double roll, double pitch, bool leftLegFlag);

    ForwardMappingResult forwardKinematics(const Eigen::Vector2d &thetaRef, bool leftLegFlag);

    void getDecoupleQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag);
    void getForwardQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag);
    std::map<bool, Eigen::Vector2d> last_solution_;
};