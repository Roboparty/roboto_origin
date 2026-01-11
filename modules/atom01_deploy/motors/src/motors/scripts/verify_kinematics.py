import numpy as np

def inverse_kinematics_py(q_roll, q_pitch, leftLegFlag):
    """
    Python implementation of the inverse_kinematics function.
    """
    l_bar = 20.0
    l_rod = np.array([180.0, 110.0])
    l_spacing = 42.35 if leftLegFlag else -42.35

    short_link_angle_0 = 180 * np.pi / 180
    long_link_angle_0 = 0 * np.pi / 180

    r_B1_0_x = -l_bar * np.cos(long_link_angle_0)
    r_B1_0_z = 180 - l_bar * np.sin(long_link_angle_0)
    r_B2_0_x = -l_bar * np.cos(short_link_angle_0)
    r_B2_0_z = 110 - l_bar * np.sin(short_link_angle_0)

    r_A_0 = [
        np.array([0, l_spacing, 180]),
        np.array([0, l_spacing, 110])
    ]
    r_B_0 = [
        np.array([r_B1_0_x, l_spacing, r_B1_0_z]),
        np.array([r_B2_0_x, l_spacing, r_B2_0_z])
    ]
    r_C_0 = [
        np.array([-20, l_spacing, 0]),
        np.array([20, l_spacing, 0])
    ]

    R_y = np.array([
        [np.cos(q_pitch), 0, np.sin(q_pitch)],
        [0, 1, 0],
        [-np.sin(q_pitch), 0, np.cos(q_pitch)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(q_roll), -np.sin(q_roll)],
        [0, np.sin(q_roll), np.cos(q_roll)]
    ])
    x_rot = R_y @ R_x

    result = {
        'r_A': [], 'r_B': [], 'r_C': [],
        'r_bar': [], 'r_rod': [], 'THETA': np.zeros(2)
    }

    for i in range(2):
        r_A_i = r_A_0[i]
        r_C_i = x_rot @ r_C_0[i]
        rBA_bar = r_B_0[i] - r_A_0[i]

        a = r_C_i[0] - r_A_i[0]
        b = r_A_i[2] - r_C_i[2]
        c = (l_rod[i]**2 - l_bar**2 - np.sum((r_C_i - r_A_i)**2)) / (2 * l_bar)

        a_sq = a**2
        b_sq = b**2
        c_sq = c**2
        ab_sq_sum = a_sq + b_sq
        
        discriminant = b_sq * c_sq - ab_sq_sum * (c_sq - a_sq)
        if discriminant < 0:
            print(f"Warning: Negative discriminant in inverse kinematics for leg {i}. Setting to 0.")
            discriminant = 0
        
        # Ensure the argument to asin is within [-1, 1]
        asin_arg = (b * c + np.sqrt(discriminant)) / ab_sq_sum
        asin_arg = np.clip(asin_arg, -1.0, 1.0)

        theta_i = np.arcsin(asin_arg)
        theta_i = theta_i if a < 0 else -theta_i

        R_y_theta = np.array([
            [np.cos(theta_i), 0, np.sin(theta_i)],
            [0, 1, 0],
            [-np.sin(theta_i), 0, np.cos(theta_i)]
        ])

        r_B_i = r_A_i + R_y_theta @ rBA_bar
        r_bar_i = r_B_i - r_A_i
        r_rod_i = r_C_i - r_B_i

        result['r_A'].append(r_A_i)
        result['r_B'].append(r_B_i)
        result['r_C'].append(r_C_i)
        result['r_bar'].append(r_bar_i)
        result['r_rod'].append(r_rod_i)
        result['THETA'][i] = theta_i

    return result

def jacobian_py(r_C, r_bar, r_rod, q_pitch):
    """
    Python implementation of the jacobian function.
    """
    s_unit = np.array([0, 1, 0])
    
    J_x = np.zeros((2, 6))
    J_x[0, 0:3] = r_rod[0].T
    J_x[0, 3:6] = np.cross(r_C[0], r_rod[0]).T
    J_x[1, 0:3] = r_rod[1].T
    J_x[1, 3:6] = np.cross(r_C[1], r_rod[1]).T

    J_theta = np.zeros((2, 2))
    J_theta[0, 0] = s_unit.dot(np.cross(r_bar[0], r_rod[0]))
    J_theta[1, 1] = s_unit.dot(np.cross(r_bar[1], r_rod[1]))

    J_q = np.zeros((6, 2))
    J_q[3, 1] = np.cos(q_pitch)
    J_q[4, 0] = 1
    J_q[5, 1] = -np.sin(q_pitch)

    J_Temp = J_x @ J_q
    
    J_motor2Joint = np.linalg.solve(J_Temp, J_theta)
    
    J_Joint2motor = np.linalg.solve(J_theta, J_Temp)

    return [J_motor2Joint, J_Joint2motor]

def verify_conversion(roll_deg, pitch_deg, joint_vel, joint_tau, leftLegFlag):
    """
    Verifies that forward kinematics can correctly reverse the inverse kinematics.
    Also verifies velocity and torque transformations.
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    leg_str = "Left" if leftLegFlag else "Right"
    
    print(f"\n----- Verifying {leg_str} Leg -----")
    print(f"Initial Ankle Joint -> Roll: {roll_deg:.4f} deg, Pitch: {pitch_deg:.4f} deg")

    # 1. Inverse Kinematics (Ankle Joint -> Motor)
    ik_result = inverse_kinematics_py(roll, pitch, leftLegFlag)
    theta_target = ik_result['THETA']
    print(f"Calculated Motor Angles (rad) -> Theta1: {theta_target[0]:.4f}, Theta2: {theta_target[1]:.4f}")

    # Get Jacobians from the initial state
    Jac_list_initial = jacobian_py(ik_result['r_C'], ik_result['r_bar'], ik_result['r_rod'], pitch)
    J_motor2Joint_initial = Jac_list_initial[0]
    J_Joint2motor_initial = Jac_list_initial[1]

    # 2. Forward Kinematics (Motor -> Ankle Joint) using Newton-Raphson
    print("--- Running Forward Kinematics (iterative solver) ---")
    
    # Initial guess for [pitch, roll] in radians
    x_c_k = np.array([0.0, 0.0]) 
    
    count = 0
    MAX_ITERATIONS = 100
    TOLERANCE = 1e-3
    ALPHA = 0.5
    
    J_motor2Joint_final = None
    J_Joint2motor_final = None

    for i in range(MAX_ITERATIONS):
        count = i + 1
        current_pitch, current_roll = x_c_k[0], x_c_k[1]

        kinematics = inverse_kinematics_py(current_roll, current_pitch, leftLegFlag)
        Jac_list = jacobian_py(kinematics['r_C'], kinematics['r_bar'], kinematics['r_rod'], current_pitch)
        
        J_motor2Joint_final = Jac_list[0]
        J_Joint2motor_final = Jac_list[1]
        
        f_error = theta_target - kinematics['THETA']
        
        if np.linalg.norm(f_error) < TOLERANCE:
            break

        x_c_k = x_c_k + ALPHA * J_motor2Joint_final @ f_error

    # 3. Compare position results
    final_pitch_rad, final_roll_rad = x_c_k[0], x_c_k[1]
    final_pitch_deg, final_roll_deg = np.rad2deg(final_pitch_rad), np.rad2deg(final_roll_rad)

    print(f"Converged in {count} iterations.")
    print(f"Final Ankle Joint -> Roll: {final_roll_deg:.4f} deg, Pitch: {final_pitch_deg:.4f} deg")

    error_roll = abs(final_roll_deg - roll_deg)
    error_pitch = abs(final_pitch_deg - pitch_deg)
    print(f"Position Error -> Roll: {error_roll:.6f} deg, Pitch: {error_pitch:.6f} deg")

    position_consistent = error_roll < 1e-1 and error_pitch < 1e-1
    if position_consistent:
        print("SUCCESS: Forward and inverse kinematics are consistent.")
    else:
        print("FAILURE: Discrepancy found in position kinematics.")

    # 4. Verify Velocity and Torque transformations
    print("\n--- Verifying Velocity and Torque Transformations ---")
    
    # Verify that the jacobians are inverses of each other
    identity_check = J_motor2Joint_final @ J_Joint2motor_final
    jacobian_inverse_ok = np.allclose(identity_check, np.identity(2))
    print(f"Jacobian inverse check (J_m2j @ J_j2m == I): {jacobian_inverse_ok}")
    if not jacobian_inverse_ok:
        print("WARNING: Jacobians are not perfect inverses.")
        print(identity_check)

    print(f"Initial Joint Vel: {joint_vel}, Initial Joint Tau: {joint_tau}")

    # Joint -> Motor
    motor_vel = J_Joint2motor_final @ joint_vel
    motor_tau = J_motor2Joint_final.T @ joint_tau
    print(f"Calculated Motor Vel: {motor_vel}, Calculated Motor Tau: {motor_tau}")

    # Motor -> Joint
    recalc_joint_vel = J_motor2Joint_final @ motor_vel
    recalc_joint_tau = J_Joint2motor_final.T @ motor_tau
    print(f"Recalculated Joint Vel: {recalc_joint_vel}, Recalculated Joint Tau: {recalc_joint_tau}")

    # Check errors
    vel_error = np.linalg.norm(joint_vel - recalc_joint_vel)
    tau_error = np.linalg.norm(joint_tau - recalc_joint_tau)
    print(f"Transformation Error -> Vel: {vel_error:.6f}, Tau: {tau_error:.6f}")

    transform_consistent = vel_error < 1e-6 and tau_error < 1e-6
    if transform_consistent:
        print("SUCCESS: Velocity and torque transformations are consistent.")
    else:
        print("FAILURE: Discrepancy found in velocity/torque transformations.")

    if position_consistent and transform_consistent:
        print("\nOverall Result: SUCCESS")
    else:
        print("\nOverall Result: FAILURE")


if __name__ == '__main__':
    verify_conversion(roll_deg=10.0, pitch_deg=0.0, joint_vel=np.array([0.0, 0.0]), joint_tau=np.array([0.0, -10.0]), leftLegFlag=True)