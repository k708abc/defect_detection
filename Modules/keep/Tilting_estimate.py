import math
import matplotlib.pyplot as plt
import numpy as np

# Function to return the rotated position
def rotation(cood, rot_angle):
    return [
        cood[0] * math.cos(rot_angle) - cood[1] * math.sin(rot_angle),
        cood[0] * math.sin(rot_angle) + cood[1] * math.cos(rot_angle),
    ]


def tilt_estimate_in(lobe_B_or, lobe_C_or, lobe_D_or, Fe_or):
    # Initial inputs2 (others)
    steps = 200  # How many angles to calculate
    alpha_range = math.pi / 2  # fitting range
    output_order = 2  # the number of digits after the diceimal point
    #
    # Values from input
    O_or = [(lobe_C_or[0] + lobe_D_or[0]) / 2, (lobe_C_or[1] + lobe_D_or[1]) / 2]
    OC = math.sqrt((O_or[0] - lobe_C_or[0]) ** 2 + (O_or[1] - lobe_C_or[1]) ** 2)
    OB = math.sqrt((O_or[0] - lobe_B_or[0]) ** 2 + (O_or[1] - lobe_B_or[1]) ** 2)
    BC = math.sqrt(
        (lobe_B_or[0] - lobe_C_or[0]) ** 2 + (lobe_B_or[1] - lobe_C_or[1]) ** 2
    )
    #
    # Prepareing theta set
    max_alpha = alpha_range
    min_alpha = -alpha_range
    diff = (max_alpha - min_alpha) / steps
    alpha_list = [min_alpha + diff * i for i in range(0, steps)]
    #
    # List for recording the values
    beta_list = []
    t_list = []
    diviation_list = []
    #
    # theta loop
    for alpha in alpha_list:
        t = OC / math.cos(alpha)
        if t > OB:
            val1 = -math.sqrt((1 - (OB / t) ** 2)) / math.cos(alpha)
            if abs(val1) <= 1:
                beta = math.asin(val1)
                BC_estimate = t * math.sqrt(
                    (math.cos(beta)) ** 2
                    + (math.sin(alpha) * math.sin(beta) - math.cos(alpha)) ** 2
                )
                error = abs(BC - BC_estimate)
                #
                beta_list.append(beta)
                t_list.append(t)
                diviation_list.append(error)
            else:
                beta_list.append(100)
                t_list.append(False)
                diviation_list.append(10)
        else:
            beta_list.append(100)
            t_list.append(False)
            diviation_list.append(10)
    #
    # Get best fit values
    best_index = diviation_list.index(min(diviation_list))
    best_alpha = alpha_list[best_index]
    best_beta = beta_list[best_index]
    best_t = t_list[best_index]
    #
    # Calculate positions from the best fit values
    A = [
        best_t * math.cos(best_beta),
        -best_t * math.sin(best_alpha) * math.sin(best_beta),
    ]
    B = [
        -best_t * math.cos(best_beta),
        best_t * math.sin(best_alpha) * math.sin(best_beta),
    ]
    C = (0, best_t * math.cos(best_alpha))
    D = (0, -best_t * math.cos(best_alpha))

    #
    # shift experimental O position to be [0,0]
    lobe_B = [x - y for (x, y) in zip(lobe_B_or, O_or)]
    lobe_C = [x - y for (x, y) in zip(lobe_C_or, O_or)]
    lobe_D = [x - y for (x, y) in zip(lobe_D_or, O_or)]
    Fe = [x - y for (x, y) in zip(Fe_or, O_or)]
    O = [0, 0]
    # rotate experiment to fit with simulation (O-lobe C angle to be 0 degree)
    rot_angle = math.atan2(lobe_C[0] - O[0], lobe_C[1] - O[1])
    lobe_B = rotation(lobe_B, rot_angle)
    lobe_C = rotation(lobe_C, rot_angle)
    lobe_D = rotation(lobe_D, rot_angle)
    Fe = rotation(Fe, rot_angle)

    # image_test(lobe_B, lobe_C, lobe_D, Fe)
    #
    # calculate best fit h
    # prepareing h set
    h_list = [i / 100 for i in range(1, 100)]
    # prepareing a list for recording error
    div_h = []
    # h loop
    for h in h_list:
        T = (
            -h * math.sin(best_beta),
            -h * math.sin(best_alpha) * math.cos(best_beta),
        )
        div = math.sqrt((T[0] - Fe[0]) ** 2 + (T[1] - Fe[1]) ** 2)
        div_h.append(div)
    # getting best fit h
    best_index_h = div_h.index(min(div_h))
    best_h = h_list[best_index_h]
    # calculate the position of Fe (T) after the rotation
    T = (
        -best_h * math.sin(best_beta),
        -best_h * math.sin(best_alpha) * math.cos(best_beta),
    )

    A = rotation(A, -rot_angle)
    B = rotation(B, -rot_angle)
    C = rotation(C, -rot_angle)
    D = rotation(D, -rot_angle)
    T = rotation(T, -rot_angle)
    #
    A = [x + y for (x, y) in zip(A, O_or)]
    B = [x + y for (x, y) in zip(B, O_or)]
    C = [x + y for (x, y) in zip(C, O_or)]
    D = [x + y for (x, y) in zip(D, O_or)]
    T = [x + y for (x, y) in zip(T, O_or)]
    Fe_3D = (
        -math.sin(best_beta) * best_h,
        -math.sin(best_alpha) * math.cos(best_beta) * best_h,
        math.cos(best_alpha) * math.cos(best_beta) * best_h,
    )
    # Organaize the values for Figure
    best_alpha = round(best_alpha * 180 / math.pi, output_order)
    best_beta = round(best_beta * 180 / math.pi, output_order)
    best_t = round(best_t, output_order)
    best_h = round(best_h, output_order)
    error = min(div_h)

    return A, B, C, D, T, best_alpha, best_beta, best_t, best_h, error, Fe_3D


def cal_error(pos_1, pos_2):
    error = (pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2
    return error


def tilt_estimate_in_new(lobe_B_or, lobe_C_or, lobe_D_or, Fe_or):
    steps_angle = 100
    angle_range = math.pi / 3
    steps_t = 50
    max_t = 0.8
    min_t = 0.3
    steps_h = 40
    max_h = 0.5
    min_h = 0.1
    max_angle = angle_range
    min_angle = -angle_range
    diff = (max_angle - min_angle) / steps_angle
    alpha_list = [min_angle + diff * i for i in range(0, steps_angle)]
    beta_list = [min_angle + diff * i for i in range(0, steps_angle)]
    diff = (max_t - min_t) / steps_t
    t_list = [min_t + diff * i for i in range(0, steps_t)]
    diff = (max_h - min_h) / steps_h
    h_list = [min_h + diff * i for i in range(0, steps_h)]
    error_rec = 100000000000
    loop_num = 1
    for alpha in alpha_list:
        print(str(loop_num) + "/" + str(len(alpha_list)))
        loop_num += 1
        for beta in beta_list:
            for t in t_list:
                A = [
                    t * math.cos(beta),
                    (-1) * t * math.sin(alpha) * math.sin(beta),
                ]
                B = [
                    (-1) * t * math.cos(beta),
                    t * math.sin(alpha) * math.sin(beta),
                ]
                C = [0, t * math.cos(alpha)]
                D = [0, (-1) * t * math.cos(alpha)]
                error_B = cal_error(B, lobe_B_or)
                error_C = cal_error(C, lobe_C_or)
                error_D = cal_error(D, lobe_D_or)
                error_BCD = error_B + error_C + error_D
                prev_error = 10000000
                prev_h = 0
                for h in h_list:
                    T = [
                        (-1) * h * math.sin(beta),
                        (-1) * h * math.sin(alpha) * math.cos(beta),
                        h * math.cos(alpha) * math.cos(beta),
                    ]
                    error_T = cal_error(T, Fe_or)
                    if error_T < prev_error:
                        prev_error = error_T
                        prev_h = h
                    else:
                        error_total = math.sqrt(error_BCD + prev_error)
                        if error_total < error_rec:
                            error_rec = error_total
                            alpha_rec = alpha
                            beta_rec = beta
                            t_rec = t
                            h_rec = prev_h
                            A_rec = A
                            B_rec = B
                            C_rec = C
                            D_rec = D
                            T_rec = T
                        break
    alpha_rec = round(alpha_rec * 180 / math.pi, 2)
    beta_rec = round(beta_rec * 180 / math.pi, 2)
    return (
        A_rec,
        B_rec,
        C_rec,
        D_rec,
        [T_rec[0], T_rec[1]],
        alpha_rec,
        beta_rec,
        round(t_rec, 2),
        round(h_rec, 2),
        round(error_rec, 2),
        T_rec,
    )


def polar_conversion(O, Fe_3D):
    theta = round(math.atan2(Fe_3D[1], Fe_3D[0]) * 180 / math.pi, 2)
    phi = round(
        (math.acos(Fe_3D[2] / math.sqrt(Fe_3D[0] ** 2 + Fe_3D[1] ** 2 + Fe_3D[2] ** 2)))
        * 180
        / math.pi,
        2,
    )
    return theta, phi


def image_test(lobe_B, lobe_C, lobe_D, Fe):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(lobe_B[0], lobe_B[1], label="Lobe B", color="red", marker="x")
    plt.scatter(lobe_C[0], lobe_C[1], label="Lobe C", color="blue", marker="x")
    plt.scatter(lobe_D[0], lobe_D[1], label="Lobe D", color="green", marker="x")
    plt.scatter(Fe[0], Fe[1], label="Lobe Fe", color="red", marker="o")
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.legend()
    ax.set_aspect("equal")
    plt.axis("square")
    plt.show()


def record_each_data(
    lobe_B,
    lobe_C,
    lobe_D,
    Fe,
    O,
    B,
    C,
    D,
    T,
    best_alpha,
    best_beta,
    best_t,
    best_h,
    error,
    theta,
    phi,
    base_name,
    count,
):

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(
        [lobe_B[0], lobe_C[0], lobe_D[0], Fe[0]],
        [lobe_B[1], lobe_C[1], lobe_D[1], Fe[1]],
        label="Experiment",
        color="red",
        marker="x",
    )
    ax.scatter(
        [B[0], C[0], D[0], T[0]],
        [B[1], C[1], D[1], T[1]],
        label="Estimate",
        color="blue",
        marker="o",
        facecolor="None",
    )

    ax.scatter([O[0]], [O[1]], marker=".")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_aspect("equal")
    ax.axis("square")
    ax.set_title("Terrace")
    #
    dx = fig.add_subplot(1, 2, 2)
    dx.scatter([], [], marker="x", color="red", label="Experiment")
    dx.scatter([], [], marker="o", color="blue", label="Estimate")
    dx.scatter(
        [],
        [],
        marker=" ",
        label="alpha = " + str(best_alpha),
    )
    dx.scatter(
        [],
        [],
        marker=" ",
        label="beta = " + str(best_beta),
    )
    dx.scatter(
        [],
        [],
        marker=" ",
        label="t = " + str(best_t),
    )
    dx.scatter(
        [],
        [],
        marker=" ",
        label="h = " + str(best_h),
    )
    dx.scatter(
        [],
        [],
        marker=" ",
        label="theta = " + str(theta),
    )
    dx.scatter(
        [],
        [],
        marker=" ",
        label="phi = " + str(phi),
    )
    dx.scatter(
        [],
        [],
        marker=" ",
        label="error = " + str(error),
    )
    dx.legend()
    dx.axis("off")
    dx.set_aspect("equal")
    rec_name = base_name + str(count) + ".png"
    plt.savefig(rec_name)
    plt.cla()
    plt.close(fig)
    rec_name = base_name + str(count) + ".txt"
    rec_text(
        rec_name,
        lobe_B,
        lobe_C,
        lobe_D,
        Fe,
        O,
        B,
        C,
        D,
        T,
        best_alpha,
        best_beta,
        best_t,
        best_h,
        error,
        theta,
        phi,
    )


def rec_text(
    rec_name,
    lobe_B,
    lobe_C,
    lobe_D,
    Fe,
    O,
    B,
    C,
    D,
    T,
    best_alpha,
    best_beta,
    best_t,
    best_h,
    error,
    theta,
    phi,
):
    with open(rec_name, mode="w") as f:
        f.write(rec_name + "\n\n")
        f.write("Experimental data ponts:" + "\n")
        f.write("lobe_B:" + "\t" + str(lobe_B[0]) + ", " + str(lobe_B[1]) + "\n")
        f.write("lobe_C:" + "\t" + str(lobe_C[0]) + ", " + str(lobe_C[1]) + "\n")
        f.write("lobe_D:" + "\t" + str(lobe_D[0]) + ", " + str(lobe_D[1]) + "\n")
        f.write("Fe:" + "\t" + str(Fe[0]) + ", " + str(Fe[1]) + "\n\n")
        f.write("Simulated ponts:" + "\n")
        f.write("lobe_B:" + "\t" + str(B[0]) + ", " + str(B[1]) + "\n")
        f.write("lobe_C:" + "\t" + str(C[0]) + ", " + str(C[1]) + "\n")
        f.write("lobe_D:" + "\t" + str(D[0]) + ", " + str(D[1]) + "\n")
        f.write("T:" + "\t" + str(T[0]) + ", " + str(T[1]) + "\n\n")
        f.write("O:" + "\t" + str(O[0]) + ", " + str(O[1]) + "\n\n")
        f.write("Simulated results:" + "\n")
        f.write("Alpha:" + "\t" + str(best_alpha) + "\n")
        f.write("Beta:" + "\t" + str(best_beta) + "\n")
        f.write("t:" + "\t" + str(best_t) + "\n")
        f.write("h:" + "\t" + str(best_h) + "\n")
        f.write("error:" + "\t" + str(error) + "\n")
        f.write("Theta:" + "\t" + str(theta) + "\n")
        f.write("Phi:" + "\t" + str(phi) + "\n")


def tilt_estimate(
    Fe_x, Fe_y, lobeB_x, lobeB_y, lobeC_x, lobeC_y, lobeD_x, lobeD_y, base_name
):
    alpha_list = []
    beta_list = []
    theta_list = []
    phi_list = []
    t_list = []
    h_list = []
    error_list = []
    A_list = []
    B_list = []
    C_list = []
    D_list = []
    T_list = []
    count = 0
    for i in range(len(Fe_x)):
        lobe_B = [lobeB_x[i], lobeB_y[i]]
        lobe_C = [lobeC_x[i], lobeC_y[i]]
        lobe_D = [lobeD_x[i], lobeD_y[i]]
        Fe = [Fe_x[i], Fe_y[i]]
        count += 1

        if None in (lobe_B + lobe_C + lobe_D + Fe):
            pass
        else:
            (
                A,
                B,
                C,
                D,
                T,
                best_alpha,
                best_beta,
                best_t,
                best_h,
                error,
                Fe_3D,
            ) = tilt_estimate_in(lobe_B, lobe_C, lobe_D, Fe)
            O = [(C[0] + D[0]) / 2, (C[1] + D[1]) / 2, 0]
            theta, phi = polar_conversion(O, Fe_3D)
            alpha_list.append(best_alpha)
            beta_list.append(best_beta)
            theta_list.append(theta)
            phi_list.append(phi)
            t_list.append(best_t)
            h_list.append(best_h)
            error_list.append(error)
            #
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)
            D_list.append(D)
            T_list.append(T)
            record_each_data(
                lobe_B,
                lobe_C,
                lobe_D,
                Fe,
                O,
                B,
                C,
                D,
                T,
                best_alpha,
                best_beta,
                best_t,
                best_h,
                error,
                theta,
                phi,
                base_name,
                count,
            )
    A_ave = get_average(A_list)
    B_ave = get_average(B_list)
    C_ave = get_average(C_list)
    D_ave = get_average(D_list)
    T_ave = get_average(T_list)
    ave_pos_x = [B_ave[0], C_ave[0], D_ave[0], T_ave[0]]
    ave_pos_y = [B_ave[1], C_ave[1], D_ave[1], T_ave[1]]
    O = [(ave_pos_x[1] + ave_pos_x[2]) / 2, (ave_pos_y[1] + ave_pos_y[2]) / 2]
    return (
        alpha_list,
        beta_list,
        theta_list,
        phi_list,
        t_list,
        h_list,
        error_list,
        ave_pos_x,
        ave_pos_y,
        O,
    )


def adjust_position(lobe_B, lobe_C, lobe_D, Fe):
    O_or = [(lobe_C[0] + lobe_D[0]) / 2, (lobe_C[1] + lobe_D[1]) / 2]
    lobe_B = [x - y for (x, y) in zip(lobe_B, O_or)]
    lobe_C = [x - y for (x, y) in zip(lobe_C, O_or)]
    lobe_D = [x - y for (x, y) in zip(lobe_D, O_or)]
    Fe = [x - y for (x, y) in zip(Fe, O_or)]
    O = [0, 0]
    # rotate experiment to fit with simulation (O-lobe C angle to be 0 degree)
    rot_angle = math.atan2(lobe_C[0] - O[0], lobe_C[1] - O[1])
    lobe_B = rotation(lobe_B, rot_angle)
    lobe_C = rotation(lobe_C, rot_angle)
    lobe_D = rotation(lobe_D, rot_angle)
    Fe = rotation(Fe, rot_angle)
    return lobe_B, lobe_C, lobe_D, Fe


def new_tilt_estimate(
    Fe_x, Fe_y, lobeB_x, lobeB_y, lobeC_x, lobeC_y, lobeD_x, lobeD_y, base_name
):
    alpha_list = []
    beta_list = []
    theta_list = []
    phi_list = []
    t_list = []
    h_list = []
    error_list = []
    A_list = []
    B_list = []
    C_list = []
    D_list = []
    T_list = []
    count = 0
    for i in range(len(Fe_x)):
        count += 1
        lobe_B = [lobeB_x[i], lobeB_y[i]]
        lobe_C = [lobeC_x[i], lobeC_y[i]]
        lobe_D = [lobeD_x[i], lobeD_y[i]]
        Fe = [Fe_x[i], Fe_y[i]]
        lobe_B, lobe_C, lobe_D, Fe = adjust_position(lobe_B, lobe_C, lobe_D, Fe)

        if None in (lobe_B + lobe_C + lobe_D + Fe):
            pass
        else:
            (
                A,
                B,
                C,
                D,
                T,
                best_alpha,
                best_beta,
                best_t,
                best_h,
                error,
                Fe_3D,
            ) = tilt_estimate_in_new(lobe_B, lobe_C, lobe_D, Fe)
            O = [(C[0] + D[0]) / 2, (C[1] + D[1]) / 2, 0]
            theta, phi = polar_conversion(O, Fe_3D)
            alpha_list.append(best_alpha)
            beta_list.append(best_beta)
            theta_list.append(theta)
            phi_list.append(phi)
            t_list.append(best_t)
            h_list.append(best_h)
            error_list.append(error)
            #
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)
            D_list.append(D)
            T_list.append(T)
            #
            record_each_data(
                lobe_B,
                lobe_C,
                lobe_D,
                Fe,
                O,
                B,
                C,
                D,
                T,
                best_alpha,
                best_beta,
                best_t,
                best_h,
                error,
                theta,
                phi,
                base_name,
                count,
            )
    A_ave = get_average(A_list)
    B_ave = get_average(B_list)
    C_ave = get_average(C_list)
    D_ave = get_average(D_list)
    T_ave = get_average(T_list)
    ave_pos_x = [B_ave[0], C_ave[0], D_ave[0], T_ave[0]]
    ave_pos_y = [B_ave[1], C_ave[1], D_ave[1], T_ave[1]]
    O = [(ave_pos_x[1] + ave_pos_x[2]) / 2, (ave_pos_y[1] + ave_pos_y[2]) / 2]

    return (
        alpha_list,
        beta_list,
        theta_list,
        phi_list,
        t_list,
        h_list,
        error_list,
        ave_pos_x,
        ave_pos_y,
        O,
    )


def get_average(values):
    x = 0
    y = 0
    for i in range(len(values)):
        x += values[i][0]
        y += values[i][1]
    x_ave = x / len(values)
    y_ave = y / len(values)
    return [x_ave, y_ave]


def get_ave_std(values):
    val_ave = sum(values) / len(values)
    val_std = np.std(values)
    return round(val_ave, 2), round(val_std, 2)
