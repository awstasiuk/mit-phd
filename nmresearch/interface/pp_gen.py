import numpy as np


def to_str(lst):
    return [str(val) for val in lst]


class PulseProgram:
    r"""
    A class which let's you generate pulse programs and phase programs and such.
    """

    @staticmethod
    def dtc_ken16(theta, fc, n_max=50):
        r"""
        Generates 6 phase programs for computing time evolution under Ken16 with arbitrary
        Rx(theta) kicking after every Ken16 Floquet sequence. 4 phase programs for Ken16,
        2 for the Rx(theta) kicking
        """

        def ph20x(ang):
            return [((270 - k * ang) % 360) for k in range(n_max)]

        def ph21x(ang):
            return [((90 - (k + 1) * ang) % 360) for k in range(n_max)]

        def ph1x(ang, prog):
            return [(phi - (k + 1) * ang) % 360 for k in range(n_max) for phi in prog]

        def to_str(lst):
            return [str(val) for val in lst]

        ph10 = [
            (phi + (i * fc + int(i / 4) * 14 * fc)) % 360
            for i, phi in enumerate(ph1x(theta, [0, 90, 90, 0]))
        ]
        ph11 = [
            (phi + ((i + 4) * fc + int(i / 4) * 14 * fc)) % 360
            for i, phi in enumerate(ph1x(theta, [0, 90, 90, 0]))
        ]
        ph12 = [
            (phi + ((i + 8) * fc + int(i / 4) * 14 * fc)) % 360
            for i, phi in enumerate(ph1x(theta, [180, 270, 270, 180]))
        ]
        ph13 = [
            (phi + ((i + 12) * fc + int(i / 4) * 14 * fc)) % 360
            for i, phi in enumerate(ph1x(theta, [180, 270, 270, 180]))
        ]
        ph20 = [
            (phi + ((16 + i * 18) * fc)) % 360 for i, phi in enumerate(ph20x(theta))
        ]
        ph21 = [
            (phi + ((17 + i * 18) * fc)) % 360 for i, phi in enumerate(ph21x(theta))
        ]

        str_list = []
        str_list.append("ph10 = (360) " + " ".join(to_str(ph10)))
        str_list.append("ph11 = (360) " + " ".join(to_str(ph11)))
        str_list.append("ph12 = (360) " + " ".join(to_str(ph12)))
        str_list.append("ph13 = (360) " + " ".join(to_str(ph13)))

        str_list.append("ph20 = (360) " + " ".join(to_str(ph20)))
        str_list.append("ph21 = (360) " + " ".join(to_str(ph21)))
        return f"\n".join(str_list)

    @staticmethod
    def dtc_ken16_compiled(theta, fc, n_max=50, swap_xy=False):
        if not swap_xy:
            ken16 = [
                [0, 0, 180, 180],
                [90, 90, 270, 270],
                [90, 90, 270, 270],
                [0, 0, 180, 180],
            ]
        else:
            ken16 = [
                [90, 90, 270, 270],
                [0, 0, 180, 180],
                [0, 0, 180, 180],
                [90, 90, 270, 270],
            ]

        shifts = [
            [fc * (4 * i + k) - int(i / 4) * theta for i in range(n_max * 4)]
            for k in range(4)
        ]

        for k in range(n_max - 1):
            shifts[3][3 + 4 * k] -= theta

        ken16_shifted = [
            [(ken16[k][idx % 4] + shift) % 360 for idx, shift in enumerate(shifts[k])]
            for k in range(4)
        ]

        str_list = []

        str_list.append("ph0 = (360) " + " ".join(to_str(ken16_shifted[0])))
        str_list.append("ph1 = (360) " + " ".join(to_str(ken16_shifted[1])))
        str_list.append("ph2 = (360) " + " ".join(to_str(ken16_shifted[2])))
        str_list.append("ph3 = (360) " + " ".join(to_str(ken16_shifted[3])))

        return f"\n".join(str_list)

    @staticmethod
    def dtc_ken16_compiled_8(theta, fc, n_max=100, phi=0):
        r"""
        Ken16 phase programs written over 8 phase programs, allowing for more than 100
        Floquet cycles. Framechange is supported. An arbitrary Ry(theta) pulse is compiled
        into the end of each phase program (set to 0 for vanilla Ken16). An optional and
        additional Z kicking is available for the start of each cycle, allowing for
        assymetric Trotterized Z fields.
        """
        ken16 = [
            [0, 180],
            [90, 270],
            [90, 270],
            [0, 180],
            [0, 180],
            [90, 270],
            [90, 270],
            [0, 180],
        ]

        shifts = [
            [
                fc * (8 * i + k) - int(i / 2) * (theta + phi) - phi
                for i in range(n_max * 2)
            ]
            for k in range(8)
        ]

        for k in range(n_max - 1):
            shifts[7][1 + 2 * k] -= theta

        ken16_shifted = [
            [(ken16[k][idx % 2] + shift) % 360 for idx, shift in enumerate(shifts[k])]
            for k in range(8)
        ]

        str_list = []

        str_list.append("ph0 = (360) " + " ".join(to_str(ken16_shifted[0])))
        str_list.append("ph1 = (360) " + " ".join(to_str(ken16_shifted[1])))
        str_list.append("ph2 = (360) " + " ".join(to_str(ken16_shifted[2])))
        str_list.append("ph3 = (360) " + " ".join(to_str(ken16_shifted[3])))
        str_list.append("ph4 = (360) " + " ".join(to_str(ken16_shifted[4])))
        str_list.append("ph5 = (360) " + " ".join(to_str(ken16_shifted[5])))
        str_list.append("ph6 = (360) " + " ".join(to_str(ken16_shifted[6])))
        str_list.append("ph7 = (360) " + " ".join(to_str(ken16_shifted[7])))

        return f"\n".join(str_list)

    @staticmethod
    def cory48(fc, n_max=50, xx=False):
        raw = (
            np.array(
                [
                    [0, 0, 0, 3, 3, 3, 2, 2, 2, 1, 1, 1],
                    [1, 1, 3, 2, 2, 0, 1, 1, 3, 2, 2, 0],
                    [2, 0, 0, 1, 3, 3, 2, 0, 0, 1, 3, 3],
                    [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0],
                ]
            )
            * 90
        )

        c48 = raw.transpose().reshape((6, 8)).transpose()

        ph_prog_len = len(c48)
        ph_prog_depth = len(c48[0])

        shifts = [
            [fc * (ph_prog_len * i + k) for i in range(n_max * ph_prog_depth)]
            for k in range(ph_prog_len)
        ]

        c48_shifted = [
            [
                (c48[k][idx % ph_prog_depth] + shift) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]
        str_list = []

        if xx:
            str_list.append("5m ip29*" + str((48 * fc) % 360))
            str_list.append("")
            str_list.append(
                "ph28 = (360) "
                + " ".join(to_str([90 - fc, 90 - fc, 270 - fc, 270 - fc]))
            )

        for idx, phases in enumerate(c48_shifted):
            str_list.append("ph" + str(idx) + " = (360) " + " ".join(to_str(phases)))

        return f"\n".join(str_list)

    def mrev8(fc, n_max, xx=False):
        mrev = [
            [180, 0],
            [90, 90],
            [270, 270],
            [0, 180],
        ]

        ph_prog_len = len(mrev)
        ph_prog_depth = len(mrev[0])

        shifts = [
            [fc * (ph_prog_len * i + k) for i in range(n_max * ph_prog_depth)]
            for k in range(ph_prog_len)
        ]

        mrev_shifted = [
            [
                (mrev[k][idx % ph_prog_depth] + shift) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]
        str_list = []

        if xx:
            str_list.append(
                "5m ip29*" + str((shifts[-1][ph_prog_depth - 1] + fc) % 360)
            )
            str_list.append("")
            str_list.append(
                "ph28 = (360) "
                + " ".join(to_str([90 - fc, 90 - fc, 270 - fc, 270 - fc]))
            )

        for idx, phases in enumerate(mrev_shifted):
            str_list.append("ph" + str(idx) + " = (360) " + " ".join(to_str(phases)))

        return f"\n".join(str_list)

    def wahuha8(fc, n_max, xx=False):
        whh8 = [
            [0, 180],
            [90, 270],
            [270, 90],
            [180, 0],
        ]

        ph_prog_len = len(whh8)
        ph_prog_depth = len(whh8[0])

        shifts = [
            [fc * (ph_prog_len * i + k) for i in range(n_max * ph_prog_depth)]
            for k in range(ph_prog_len)
        ]

        whh8_shifted = [
            [
                (whh8[k][idx % ph_prog_depth] + shift) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]
        str_list = []

        if xx:
            str_list.append(
                "5m ip29*" + str((shifts[-1][ph_prog_depth - 1] + fc) % 360)
            )
            str_list.append("")
            str_list.append(
                "ph28 = (360) "
                + " ".join(to_str([90 - fc, 90 - fc, 270 - fc, 270 - fc]))
            )

        for idx, phases in enumerate(whh8_shifted):
            str_list.append("ph" + str(idx) + " = (360) " + " ".join(to_str(phases)))

        return f"\n".join(str_list)

    @staticmethod
    def peng24(fc, n_max=50, theta=0, xx=False):
        r"""
        Generates 12 phase programs needed to run yxx24 suspension sequence using a variable
        framechage angle and a compiled Ry(theta) pulse at the end of each sequence.
        """
        yxx24 = [
            [270, 90],
            [0, 180],
            [180, 0],
            [90, 270],
            [180, 0],
            [180, 0],
            [90, 270],
            [180, 0],
            [0, 180],
            [270, 90],
            [0, 180],
            [0, 180],
        ]

        ph_prog_len = len(yxx24)
        ph_prog_depth = len(yxx24[0])

        shifts = [
            [
                fc * (ph_prog_len * i + k) - int(i / ph_prog_depth) * theta
                for i in range(n_max * ph_prog_depth)
            ]
            for k in range(ph_prog_len)
        ]

        for k in range(n_max - 1):
            shifts[ph_prog_len - 1][ph_prog_depth * (k + 1) - 1] -= theta

        yxx24_shifted = [
            [
                (yxx24[k][idx % ph_prog_depth] + shift) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]

        str_list = []

        if xx:
            str_list.append(
                "5m ip29*" + str((shifts[-1][ph_prog_depth - 1] + fc) % 360)
            )
            str_list.append("")
            str_list.append(
                "ph28 = (360) "
                + " ".join(to_str([90 - fc, 90 - fc, 270 - fc, 270 - fc]))
            )

        for idx, phases in enumerate(yxx24_shifted):
            str_list.append("ph" + str(idx) + " = (360) " + " ".join(to_str(phases)))

        return f"\n".join(str_list)
    
    @staticmethod
    def angle12(fc, n_max=50, theta=0, xx=False):
        r"""
        Generates 12 phase programs needed to run Angle12 suspension sequence using a variable
        framechage angle and a compiled Ry(theta) pulse at the end of each sequence.
        """
        ang12 = [
            [270],
            [0],
            [180],
            [90],
            [180],
            [180],
            [90],
            [180],
            [0],
            [270],
            [0],
            [0],
        ]

        ph_prog_len = len(ang12)
        ph_prog_depth = len(ang12[0])

        shifts = [
            [
                fc * (ph_prog_len * i + k) - int(i / ph_prog_depth) * theta
                for i in range(n_max * ph_prog_depth)
            ]
            for k in range(ph_prog_len)
        ]

        for k in range(n_max - 1):
            shifts[ph_prog_len - 1][ph_prog_depth * (k + 1) - 1] -= theta

        ang12_shifted = [
            [
                (ang12[k][idx % ph_prog_depth] + shift) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]

        str_list = []

        if xx:
            str_list.append(
                "5m ip29*" + str((shifts[-1][ph_prog_depth - 1] + fc) % 360)
            )
            str_list.append("")
            str_list.append(
                "ph28 = (360) "
                + " ".join(to_str([90 - fc, 90 - fc, 270 - fc, 270 - fc]))
            )

        for idx, phases in enumerate(ang12_shifted):
            str_list.append("ph" + str(idx) + " = (360) " + " ".join(to_str(phases)))

        return f"\n".join(str_list)

    @staticmethod
    def wahuha_helper(init_phase, fc, whh_cycles=18):
        whh8 = [
            [0, 180],
            [90, 270],
            [270, 90],
            [180, 0],
        ]

        shifts = [
            [init_phase + fc * (4 * i + k) for i in range(whh_cycles * 2)]
            for k in range(4)
        ]

        whh_fc = [
            [(whh8[k][idx % 2] + shift) % 360 for idx, shift in enumerate(shifts[k])]
            for k in range(4)
        ]

        return whh_fc, (whh_cycles * 8 * fc)

    @staticmethod
    def dtc_dis_int(theta, fc, n_max=50):
        r"""
        Generates the phase programs needed to measure disordered Zeeman states evolving
        under the internal Hamiltonian with arbitrary Rx(theta) kicking. See the pp
        dtc_internal_165_disZ_fc10 as an example.
        """
        glb_ph = 0

        # first pulse
        ph0 = [270, 270, 90, 90]
        glb_ph += fc

        # first wahuha sequence (ph1,ph2,ph3,ph4)
        whh1, whh_ph1 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph1

        # recovery x pulse
        ph5 = [glb_ph % 360 for _ in range(4)]
        ph5.extend([(180 + glb_ph) % 360 for _ in range(4)])
        glb_ph += fc

        # periodic kicking loop... this one is special
        ph6 = [((glb_ph + 270 + k * (2 * fc - theta)) % 360) for k in range(n_max)]
        ph7 = [
            ((glb_ph + 90 - (k + 1) * theta + (2 * k + 1) * fc) % 360)
            for k in range(n_max)
        ]

        # inv recovery x pulse
        ph8 = [glb_ph % 360 for _ in range(8)]
        ph8.extend([(180 + glb_ph) % 360 for _ in range(8)])
        glb_ph += fc

        # second wahuha sequence (ph9,ph10,ph11,ph12)
        whh2, whh_ph2 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph2

        # shelving pulse
        ph13 = [(ang + glb_ph) % 360 for ang in [270, 270, 90, 90]]
        glb_ph += fc

        str_list = []
        for i in range(8, 14, 1):
            str_list.append("20m ip" + str(i) + "*" + str((2 * fc - theta) % 360))
        str_list.append("")
        str_list.append("ph0 = (360) " + " ".join(to_str(ph0)))
        str_list.append("")
        str_list.append("ph1 = (360) " + " ".join(to_str(whh1[0])))
        str_list.append("ph2 = (360) " + " ".join(to_str(whh1[1])))
        str_list.append("ph3 = (360) " + " ".join(to_str(whh1[2])))
        str_list.append("ph4 = (360) " + " ".join(to_str(whh1[3])))
        str_list.append("")
        str_list.append("ph5 = (360) " + " ".join(to_str(ph5)))
        str_list.append("")
        str_list.append("ph6 = (360) " + " ".join(to_str(ph6)))
        str_list.append("ph7 = (360) " + " ".join(to_str(ph7)))
        str_list.append("")
        str_list.append("ph8 = (360) " + " ".join(to_str(ph8)))
        str_list.append("")
        str_list.append("ph9 = (360) " + " ".join(to_str(whh2[0])))
        str_list.append("ph10 = (360) " + " ".join(to_str(whh2[1])))
        str_list.append("ph11 = (360) " + " ".join(to_str(whh2[2])))
        str_list.append("ph12 = (360) " + " ".join(to_str(whh2[3])))
        str_list.append("")
        str_list.append("ph13 = (360) " + " ".join(to_str(ph13)))
        str_list.append("")
        str_list.append("ph14 = 0 2 0 2")
        str_list.append("ph31 = 1 3 1 3 3 1 3 1 3 1 3 1 1 3 1 3")

        return f"\n".join(str_list)

    @staticmethod
    def ken16_compiled_8_helper(init_phase, theta, fc, n_max, phi):
        r"""
        Ken16 phase programs written over 8 phase programs, allowing for more than 100
        Floquet cycles. Framechange is supported. An arbitrary Ry(theta) pulse is compiled
        into the end of each phase program (set to 0 for vanilla Ken16). An optional and
        additional Z kicking is available for the start of each cycle, allowing for
        assymetric Trotterized Z fields.
        """
        ken16 = [
            [0, 180],
            [90, 270],
            [90, 270],
            [0, 180],
            [0, 180],
            [90, 270],
            [90, 270],
            [0, 180],
        ]

        shifts = [
            [
                init_phase + fc * (8 * i + k) - int(i / 2) * (theta + phi) - phi
                for i in range(n_max * 2)
            ]
            for k in range(8)
        ]

        for k in range(n_max - 1):
            shifts[7][1 + 2 * k] -= theta

        ken16_shifted = [
            [(ken16[k][idx % 2] + shift) % 360 for idx, shift in enumerate(shifts[k])]
            for k in range(8)
        ]

        return ken16_shifted

    @staticmethod
    def dtc_dis_ken16(theta, fc, n_max=50, phi=0):
        r"""
        Generates the phase programs needed to measure disordered Zeeman states evolving
        under the Ken16 with arbitrary Z(phi) and Ry(theta) kicking. See the pp
        dtc_ken16_170_disZ_fc9 as an example.
        """
        glb_ph = 0

        # first pulse
        ph0 = [270, 270, 90, 90]
        glb_ph += fc

        # first wahuha sequence (ph1,ph2,ph3,ph4)
        whh1, whh_ph1 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph1

        # recovery x pulse
        ph5 = [glb_ph % 360 for _ in range(4)]
        ph5.extend([(180 + glb_ph) % 360 for _ in range(4)])
        glb_ph += fc

        # ken16 loop... this one is special. ph6-ph13
        ken = PulseProgram.ken16_compiled_8_helper(glb_ph, theta, fc, n_max, phi)

        # inv recovery x pulse
        ph14 = [glb_ph % 360 for _ in range(8)]
        ph14.extend([(180 + glb_ph) % 360 for _ in range(8)])
        glb_ph += fc

        # second wahuha sequence (ph15,ph16,ph17,ph18)
        whh2, whh_ph2 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph2

        # shelving pulse
        ph19 = [(ang + glb_ph) % 360 for ang in [270, 270, 90, 90]]
        glb_ph += fc

        str_list = []
        for i in range(14, 20, 1):
            str_list.append(
                "25m ip" + str(i) + "*" + str((16 * fc - theta - phi) % 360)
            )
        str_list.append("")
        str_list.append("ph0 = (360) " + " ".join(to_str(ph0)))
        str_list.append("")
        str_list.append("ph1 = (360) " + " ".join(to_str(whh1[0])))
        str_list.append("ph2 = (360) " + " ".join(to_str(whh1[1])))
        str_list.append("ph3 = (360) " + " ".join(to_str(whh1[2])))
        str_list.append("ph4 = (360) " + " ".join(to_str(whh1[3])))
        str_list.append("")
        str_list.append("ph5 = (360) " + " ".join(to_str(ph5)))
        str_list.append("")
        str_list.append("ph6 = (360) " + " ".join(to_str(ken[0])))
        str_list.append("ph7 = (360) " + " ".join(to_str(ken[1])))
        str_list.append("ph8 = (360) " + " ".join(to_str(ken[2])))
        str_list.append("ph9 = (360) " + " ".join(to_str(ken[3])))
        str_list.append("ph10 = (360) " + " ".join(to_str(ken[4])))
        str_list.append("ph11 = (360) " + " ".join(to_str(ken[5])))
        str_list.append("ph12 = (360) " + " ".join(to_str(ken[6])))
        str_list.append("ph13 = (360) " + " ".join(to_str(ken[7])))
        str_list.append("")
        str_list.append("ph14 = (360) " + " ".join(to_str(ph14)))
        str_list.append("")
        str_list.append("ph15 = (360) " + " ".join(to_str(whh2[0])))
        str_list.append("ph16 = (360) " + " ".join(to_str(whh2[1])))
        str_list.append("ph17 = (360) " + " ".join(to_str(whh2[2])))
        str_list.append("ph18 = (360) " + " ".join(to_str(whh2[3])))
        str_list.append("")
        str_list.append("ph19 = (360) " + " ".join(to_str(ph19)))
        str_list.append("")
        str_list.append("ph20 = 0 2 0 2")
        str_list.append("ph31 = 1 3 1 3 3 1 3 1 3 1 3 1 1 3 1 3")

        return f"\n".join(str_list)

    @staticmethod
    def dtc_dis_peng24(theta, fc, n_max=50):
        r"""
        Generates the phase programs needed to measure disordered Zeeman states evolving
        under the Pang24 suspension with arbitrary Ry(theta) kicking. See the pp
        ** as an example.
        """
        glb_ph = 0

        # first pulse
        ph0 = [270, 270, 90, 90]
        glb_ph += fc

        # first wahuha sequence (ph1,ph2,ph3,ph4)
        whh1, whh_ph1 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph1

        # recovery x pulse
        ph5 = [glb_ph % 360 for _ in range(4)]
        ph5.extend([(180 + glb_ph) % 360 for _ in range(4)])
        glb_ph += fc

        # ken16 loop... this one is special. ph6-ph17
        peng = PulseProgram.peng24_compiled_12_helper(glb_ph, theta, fc, n_max)

        # inv recovery x pulse
        ph14 = [glb_ph % 360 for _ in range(8)]
        ph14.extend([(180 + glb_ph) % 360 for _ in range(8)])
        glb_ph += fc

        # second wahuha sequence (ph19-ph22)
        whh2, whh_ph2 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph2

        # shelving pulse
        ph19 = [(ang + glb_ph) % 360 for ang in [270, 270, 90, 90]]
        glb_ph += fc

        str_list = []
        for i in range(18, 24, 1):
            str_list.append("20m ip" + str(i) + "*" + str((24 * fc - theta) % 360))

        str_list.append("")
        str_list.append("ph0 = (360) " + " ".join(to_str(ph0)))
        str_list.append("")
        str_list.append("ph1 = (360) " + " ".join(to_str(whh1[0])))
        str_list.append("ph2 = (360) " + " ".join(to_str(whh1[1])))
        str_list.append("ph3 = (360) " + " ".join(to_str(whh1[2])))
        str_list.append("ph4 = (360) " + " ".join(to_str(whh1[3])))
        str_list.append("")
        str_list.append("ph5 = (360) " + " ".join(to_str(ph5)))
        str_list.append("")
        str_list.append("ph6 = (360) " + " ".join(to_str(peng[0])))
        str_list.append("ph7 = (360) " + " ".join(to_str(peng[1])))
        str_list.append("ph8 = (360) " + " ".join(to_str(peng[2])))
        str_list.append("ph9 = (360) " + " ".join(to_str(peng[3])))
        str_list.append("ph10 = (360) " + " ".join(to_str(peng[4])))
        str_list.append("ph11 = (360) " + " ".join(to_str(peng[5])))
        str_list.append("ph12 = (360) " + " ".join(to_str(peng[6])))
        str_list.append("ph13 = (360) " + " ".join(to_str(peng[7])))
        str_list.append("ph14 = (360) " + " ".join(to_str(peng[8])))
        str_list.append("ph15 = (360) " + " ".join(to_str(peng[9])))
        str_list.append("ph16 = (360) " + " ".join(to_str(peng[10])))
        str_list.append("ph17 = (360) " + " ".join(to_str(peng[11])))
        str_list.append("")
        str_list.append("ph18 = (360) " + " ".join(to_str(ph14)))
        str_list.append("")
        str_list.append("ph19 = (360) " + " ".join(to_str(whh2[0])))
        str_list.append("ph20 = (360) " + " ".join(to_str(whh2[1])))
        str_list.append("ph21 = (360) " + " ".join(to_str(whh2[2])))
        str_list.append("ph22 = (360) " + " ".join(to_str(whh2[3])))
        str_list.append("")
        str_list.append("ph23 = (360) " + " ".join(to_str(ph19)))
        str_list.append("")
        str_list.append("ph24 = 0 2 0 2")
        str_list.append("ph31 = 1 3 1 3 3 1 3 1 3 1 3 1 1 3 1 3")

        return f"\n".join(str_list)

    @staticmethod
    def peng24_compiled_12_helper(init_phase, theta, fc, n_max):
        r"""
        helper function for disordered z phase program
        """
        yxx24 = [
            [270, 90],
            [0, 180],
            [180, 0],
            [90, 270],
            [180, 0],
            [180, 0],
            [90, 270],
            [180, 0],
            [0, 180],
            [270, 90],
            [0, 180],
            [0, 180],
        ]

        shifts = [
            [
                init_phase + fc * (12 * i + k) - int(i / 2) * theta
                for i in range(n_max * 2)
            ]
            for k in range(12)
        ]

        for k in range(n_max - 1):
            shifts[11][1 + 2 * k] -= theta

        yxx24_shifted = [
            [(yxx24[k][idx % 2] + shift) % 360 for idx, shift in enumerate(shifts[k])]
            for k in range(12)
        ]

        return yxx24_shifted

    @staticmethod
    def yxx24_4_theta(theta, fc, n_max=50):
        r"""
        Generates 4 phase programs needed to run yxx24 suspension sequence using a variable
        framechage angle and a compiled Ry(theta) pulse at the end of each sequence.

        See dtc_yxx24_170_fc9 for an example pulse program built with this function.
        """
        yxx24 = [
            [270, 180, 0, 90, 0, 180],
            [0, 180, 270, 180, 0, 90],
            [180, 90, 0, 0, 270, 180],
            [90, 180, 0, 270, 0, 180],
        ]

        shifts = [
            [fc * (4 * i + k) - int(i / 6) * theta for i in range(n_max * 6)]
            for k in range(4)
        ]

        for k in range(n_max - 1):
            shifts[3][5 + 6 * k] -= theta

        yxx24_shifted = [
            [(yxx24[k][idx % 6] + shift) % 360 for idx, shift in enumerate(shifts[k])]
            for k in range(4)
        ]

        str_list = []

        str_list.append("ph1 = (360) " + " ".join(to_str(yxx24_shifted[0])))
        str_list.append("ph2 = (360) " + " ".join(to_str(yxx24_shifted[1])))
        str_list.append("ph3 = (360) " + " ".join(to_str(yxx24_shifted[2])))
        str_list.append("ph4 = (360) " + " ".join(to_str(yxx24_shifted[3])))

        return f"\n".join(str_list)

    @staticmethod
    def zx_kicking(theta, phi=0, fc=0, n_max=50):
        n_max = 150

        kick1 = [((270 - k * (theta + phi) - phi) % 360) for k in range(n_max)]
        kick2 = [((90 - (k + 1) * (theta + phi)) % 360) for k in range(n_max)]

        ph0 = [(phase + 2 * fc * idx) % 360 for idx, phase in enumerate(kick1)]
        ph1 = [(phase + fc + 2 * fc * idx) % 360 for idx, phase in enumerate(kick2)]
        str_list = []
        str_list.append("ph0 = (360) " + " ".join(to_str(ph0)))
        str_list.append("ph1 = (360) " + " ".join(to_str(ph1)))
        return f"\n".join(str_list)


    @staticmethod
    def localz_internal(fc):
        r"""
        Generates the phase programs needed to measure disordered Zeeman states evolving
        under the internal Hamiltonian with arbitrary Rx(theta) kicking. See the pp
        dtc_internal_165_disZ_fc10 as an example.
        """
        glb_ph = 0

        # first pulse
        ph0 = [270, 270, 90, 90]
        glb_ph += fc

        # first wahuha sequence (ph1,ph2,ph3,ph4)
        whh1, whh_ph1 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph1

        # recovery x pulse
        ph5 = [glb_ph % 360 for _ in range(4)]
        ph5.extend([(180 + glb_ph) % 360 for _ in range(4)])
        glb_ph += fc

        # periodic kicking loop... this one is special
        #ph6 = [((glb_ph + 270 + k * (2 * fc - theta)) % 360) for k in range(n_max)]
        #ph7 = [
        #    ((glb_ph + 90 - (k + 1) * theta + (2 * k + 1) * fc) % 360)
        #    for k in range(n_max)
        #]

        # inv recovery x pulse
        ph8 = [glb_ph % 360 for _ in range(8)]
        ph8.extend([(180 + glb_ph) % 360 for _ in range(8)])
        glb_ph += fc

        # second wahuha sequence (ph9,ph10,ph11,ph12)
        whh2, whh_ph2 = PulseProgram.wahuha_helper(glb_ph, fc)
        glb_ph += whh_ph2

        # shelving pulse
        ph13 = [(ang + glb_ph) % 360 for ang in [270, 270, 90, 90]]
        glb_ph += fc

        str_list = []
        #for i in range(8, 14, 1):
        #    str_list.append("20m ip" + str(i) + "*" + str((2 * fc - theta) % 360))
        str_list.append("")
        str_list.append("ph0 = (360) " + " ".join(to_str(ph0)))
        str_list.append("")
        str_list.append("ph1 = (360) " + " ".join(to_str(whh1[0])))
        str_list.append("ph2 = (360) " + " ".join(to_str(whh1[1])))
        str_list.append("ph3 = (360) " + " ".join(to_str(whh1[2])))
        str_list.append("ph4 = (360) " + " ".join(to_str(whh1[3])))
        str_list.append("")
        str_list.append("ph5 = (360) " + " ".join(to_str(ph5)))
        str_list.append("")
        #str_list.append("ph6 = (360) " + " ".join(to_str(ph6)))
        #str_list.append("ph7 = (360) " + " ".join(to_str(ph7)))
        #str_list.append("")
        str_list.append("ph8 = (360) " + " ".join(to_str(ph8)))
        str_list.append("")
        str_list.append("ph9 = (360) " + " ".join(to_str(whh2[0])))
        str_list.append("ph10 = (360) " + " ".join(to_str(whh2[1])))
        str_list.append("ph11 = (360) " + " ".join(to_str(whh2[2])))
        str_list.append("ph12 = (360) " + " ".join(to_str(whh2[3])))
        str_list.append("")
        str_list.append("ph13 = (360) " + " ".join(to_str(ph13)))
        str_list.append("")
        str_list.append("ph14 = 0 2 0 2")
        str_list.append("ph31 = 1 3 1 3 3 1 3 1 3 1 3 1 1 3 1 3")

        return f"\n".join(str_list)