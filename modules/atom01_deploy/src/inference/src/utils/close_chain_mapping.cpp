#include "close_chain_mapping.hpp"

//////********************print******************************//////

void Decouple::print_vector3d(const Eigen::Vector3d &vec)
{
    std::cout << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << "]";
}

void Decouple::print_kinematics_result(const InsKinematicsResult &result)
{
    std::cout << "r_A: ";
    for (const auto &vec : result.r_A)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_B: ";
    for (const auto &vec : result.r_B)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_C: ";
    for (const auto &vec : result.r_C)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_bar: ";
    for (const auto &vec : result.r_bar)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "r_rod: ";
    for (const auto &vec : result.r_rod)
    {
        print_vector3d(vec);
        std::cout << " ";
    }
    std::cout << std::endl;

    std::cout << "THETA: ";
    std::cout << result.THETA;
    std::cout << std::endl;
}
//////********************print******************************//////

//////********************inverse kinematics*****************//////
InsKinematicsResult
Decouple::inverse_kinematics(
    double q_roll,
    double q_pitch, bool leftLegFlag)
{
    InsKinematicsResult result;

    result.THETA = Eigen::Vector2d::Zero();

    double l_bar = 20; // # up

    double l_rod[2] = {180, 110}; // # long rod
    double l_spacing = leftLegFlag ? 42.35 : -42.35;  // # spacing between legs

    double short_link_angle_0 = 180 * M_PI / 180;
    double long_link_angle_0 = 0 * M_PI / 180;

    double r_B1_0_x = -l_bar * cos(long_link_angle_0);
    double r_B1_0_z = 180 - l_bar * sin(long_link_angle_0);
    double r_B2_0_x = -l_bar * cos(short_link_angle_0);
    double r_B2_0_z = 110 - l_bar * sin(short_link_angle_0);

    // Define points
    Eigen::Vector3d r_A1_0{0, l_spacing, 180};
    Eigen::Vector3d r_B1_0{r_B1_0_x, l_spacing, r_B1_0_z};
    Eigen::Vector3d r_C1_0{-20, l_spacing, 0};

    Eigen::Vector3d r_A2_0{0, l_spacing, 110};
    Eigen::Vector3d r_B2_0{r_B2_0_x, l_spacing, r_B2_0_z};
    Eigen::Vector3d r_C2_0{20, l_spacing, 0};

    std::vector<Eigen::Vector3d> r_A_0;
    r_A_0.push_back(r_A1_0);
    r_A_0.push_back(r_A2_0);

    std::vector<Eigen::Vector3d> r_B_0;
    r_B_0.push_back(r_B1_0);
    r_B_0.push_back(r_B2_0);

    std::vector<Eigen::Vector3d> r_C_0;
    r_C_0.push_back(r_C1_0);
    r_C_0.push_back(r_C2_0);

    // Rotation matrices
    Eigen::Matrix3d R_y = Eigen::Matrix3d::Zero();
    R_y << cos(q_pitch), 0, sin(q_pitch),
        0, 1, 0,
        -sin(q_pitch), 0, cos(q_pitch);

    Eigen::Matrix3d R_x = Eigen::Matrix3d::Zero();
    R_x << 1, 0, 0,
        0, cos(q_roll), -sin(q_roll),
        0, sin(q_roll), cos(q_roll);

    Eigen::Matrix3d x_rot = R_y * R_x;

    // Vectors to store results
    // std::vector<Eigen::Vector3d> results;

    for (int i = 0; i < 2; i++)
    {
        Eigen::Vector3d r_A_i = r_A_0[i];
        Eigen::Vector3d r_C_i = x_rot * r_C_0[i];
        Eigen::Vector3d rBA_bar = r_B_0[i] - r_A_0[i];

        double a = r_C_i[0] - r_A_i[0];
        double b = r_A_i[2] - r_C_i[2];
        double c = (l_rod[i] * l_rod[i] - l_bar * l_bar - (r_C_i - r_A_i).squaredNorm()) / (2 * l_bar);

        double a_sq = a * a;
        double b_sq = b * b;
        double c_sq = c * c;
        double ab_sq_sum = a_sq + b_sq;
        double discriminant = b_sq * c_sq - ab_sq_sum * (c_sq - a_sq);
        if (discriminant < 0) {
            std::cerr << "Warning: Negative discriminant in inverse kinematics. Setting theta_i to 0." << std::endl;
            discriminant = 0;
        }

        double theta_i = asin((b * c + sqrt(discriminant)) / ab_sq_sum);
        theta_i = a < 0 ? theta_i : -theta_i;

        Eigen::Matrix3d R_y_theta = Eigen::Matrix3d::Zero();
        R_y_theta << std::cos(theta_i), 0, std::sin(theta_i),
            0, 1, 0,
            -std::sin(theta_i), 0, std::cos(theta_i);

        Eigen::Vector3d r_B_i = r_A_i + R_y_theta * rBA_bar;
        Eigen::Vector3d r_bar_i = r_B_i - r_A_i;
        Eigen::Vector3d r_rod_i = r_C_i - r_B_i;

        // Populate results
        result.r_A.push_back(r_A_i);
        result.r_B.push_back(r_B_i);
        result.r_C.push_back(r_C_i);
        result.r_bar.push_back(r_bar_i);
        result.r_rod.push_back(r_rod_i);
        result.THETA[i] = theta_i;
    }

    return result;
}
//////********************inverse kinematics*****************//////

//////********************jacobian***************************//////
std::vector<Eigen::MatrixXd>
Decouple::jacobian(const std::vector<Eigen::Vector3d> &r_C,
                   const std::vector<Eigen::Vector3d> &r_bar,
                   const std::vector<Eigen::Vector3d> &r_rod,
                   double q_pitch)
{
    // Assuming r_C, r_bar, r_rod are vectors of Eigen::Vector3d with at least 2 elements each
    static const Eigen::Vector3d s_unit(0, 1, 0);
    
    Eigen::Matrix<double, 2, 6> J_x;
    J_x << r_rod[0].transpose(), (r_C[0].cross(r_rod[0])).transpose(),
           r_rod[1].transpose(), (r_C[1].cross(r_rod[1])).transpose();

    Eigen::Matrix2d J_theta;
    J_theta << s_unit.dot(r_bar[0].cross(r_rod[0])), 0,
               0, s_unit.dot(r_bar[1].cross(r_rod[1]));
    
    Eigen::Matrix<double, 6, 2> J_q;
    J_q << 0, 0,
           0, 0,
           0, 0,
           0, cos(q_pitch),
           1, 0,
           0, -sin(q_pitch);

    Eigen::Matrix2d J_Temp = J_x * J_q;
    
    Eigen::PartialPivLU<Eigen::Matrix2d> lu_decomp(J_Temp);
    Eigen::PartialPivLU<Eigen::Matrix2d> lu_theta(J_theta);
    
    std::vector<Eigen::MatrixXd> J_ankle(2);
    J_ankle[0] = lu_decomp.solve(J_theta);
    J_ankle[1] = lu_theta.solve(J_Temp);
    
    return J_ankle;
}
//////********************jacobian***************************//////

// from x to theta， from S to P
std::pair<Eigen::Vector2d, std::vector<Eigen::MatrixXd>>
Decouple::get_decouple(double roll, double pitch, bool leftLegFlag)
{
    InsKinematicsResult kinematics = inverse_kinematics(roll, pitch, leftLegFlag);
    // print_kinematics_result(kinematics);
    std::vector<Eigen::MatrixXd> Jac = jacobian(kinematics.r_C, kinematics.r_bar, kinematics.r_rod, pitch);
    return {kinematics.THETA, Jac};
}

//////********************forward kinematics*****************//////
ForwardMappingResult
Decouple::forward_kinematics(const Eigen::Vector2d &thetaRef, bool leftLegFlag)
{

    ForwardMappingResult mapping_result;

    int count = 0;
    Eigen::Vector2d f_error{10, 10};
    Eigen::Vector2d x_c_k = last_solution_.count(leftLegFlag) ? 
                            last_solution_[leftLegFlag] : 
                            Eigen::Vector2d::Zero();

    std::vector<Eigen::MatrixXd> Jac;
    static constexpr int MAX_ITERATIONS = 100;
    static constexpr double TOLERANCE = 1e-3;
    static constexpr double ALPHA = 0.5;
    /*after*/
    while (f_error.norm() > TOLERANCE && count < MAX_ITERATIONS)
    {
        InsKinematicsResult kinematics = inverse_kinematics(x_c_k[1], x_c_k[0], leftLegFlag);
        // print_kinematics_result(kinematics);

        Jac = jacobian(kinematics.r_C, kinematics.r_bar, kinematics.r_rod, x_c_k[0]);
        // std::cout << "===== count:" << count << "\n Jac: " << Jac << "\n THEAT:" << kinematics.THETA << std::endl;
        Eigen::MatrixXd J_motor2Joint = Jac[0];
        // Eigen::MatrixXd J_Joint2motor = Jac[1];
        if (J_motor2Joint.hasNaN())
        {
            std::cerr << "Decouple::forward_kinematics() Jac is nan!!" << std::endl;
            std::cerr << "  roll x_c_k[1],pitch  x_c_k[0] n!!" << x_c_k[1] << "   ---   " << x_c_k[0] << std::endl;
            mapping_result.count = -1;
            mapping_result.ankle_joint_ori = Eigen::Vector2d::Zero();
            mapping_result.Jac = Jac;
            return mapping_result; // -1 是失败的标记
        }

        f_error = thetaRef - kinematics.THETA;

        x_c_k = x_c_k + ALPHA * J_motor2Joint * f_error;
        // std::cout <<  " thetaCal: " << thetaCal << "\n f_error: " << f_error << "\n pitch_roll:" << x_c_k << std::endl;

        count++;
    }
    /*after*/

    if (f_error.norm() < TOLERANCE)
    {
        last_solution_[leftLegFlag] = x_c_k;
        // std::cout << leftLegFlag << " Converged in " << count << " iterations." << std::endl;
    }

    mapping_result.count = count;
    mapping_result.ankle_joint_ori = x_c_k;
    mapping_result.Jac = Jac;

    return mapping_result; // -1 是失败的标记
}
//////********************forward kinematics*****************//////

// from x to theta， from Serial to Parallel
// force control ,should input current pitch roll
void Decouple::get_decoupleQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag)
{
    double Pitch, Roll;
    Pitch = q[0]; // rotation axis [0 1 0]
    Roll = q[1];

    std::pair<Eigen::Vector2d, std::vector<Eigen::MatrixXd>> motor;

    motor = get_decouple(Roll, Pitch, leftLegFlag);
    q.segment<2>(0) = motor.first;
    vel.segment<2>(0) = motor.second[1] * (vel.segment<2>(0));
    tau.segment<2>(0) = motor.second[0].transpose() * (tau.segment<2>(0));
}

void Decouple::get_forwardQVT(Eigen::VectorXd &q, Eigen::VectorXd &vel, Eigen::VectorXd &tau, bool leftLegFlag)
{
    Eigen::Vector2d motor = Eigen::Vector2d::Zero(2, 1);
    motor = q.segment<2>(0);

    ForwardMappingResult joint = forward_kinematics(motor, leftLegFlag);
    q.segment<2>(0) = joint.ankle_joint_ori;
    vel.segment<2>(0) = joint.Jac[0] * (vel.segment<2>(0));             // vel transfer from motor to ankle joint
    tau.segment<2>(0) = joint.Jac[1].transpose() * (tau.segment<2>(0)); // tau transfer from motor to ankle joint
}