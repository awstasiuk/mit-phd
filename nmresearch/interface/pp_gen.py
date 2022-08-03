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
    def wahuha(init_phase, fc, whh_cycles=18):
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
        whh1, whh_ph1 = PulseProgram.wahuha(glb_ph, fc)
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
        whh2, whh_ph2 = PulseProgram.wahuha(glb_ph, fc)
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
    def yxx24_12_theta(theta, fc, n_max=50):
        r"""
        Generates 4 phase programs needed to run yxx24 suspension sequence using a variable
        framechage angle and a compiled Ry(theta) pulse at the end of each sequence.

        See dtc_yxx24_170_fc9 for an example pulse program built with this function.
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
            [fc * (12 * i + k) - int(i / 2) * theta for i in range(n_max * 2)]
            for k in range(12)
        ]

        for k in range(n_max - 1):
            shifts[11][1 + 2 * k] -= theta

        yxx24_shifted = [
            [(yxx24[k][idx % 2] + shift) % 360 for idx, shift in enumerate(shifts[k])]
            for k in range(12)
        ]

        str_list = []

        str_list.append("ph1 = (360) " + " ".join(to_str(yxx24_shifted[0])))
        str_list.append("ph2 = (360) " + " ".join(to_str(yxx24_shifted[1])))
        str_list.append("ph3 = (360) " + " ".join(to_str(yxx24_shifted[2])))
        str_list.append("ph4 = (360) " + " ".join(to_str(yxx24_shifted[3])))
        str_list.append("ph5 = (360) " + " ".join(to_str(yxx24_shifted[4])))
        str_list.append("ph6 = (360) " + " ".join(to_str(yxx24_shifted[5])))
        str_list.append("ph7 = (360) " + " ".join(to_str(yxx24_shifted[6])))
        str_list.append("ph8 = (360) " + " ".join(to_str(yxx24_shifted[7])))
        str_list.append("ph9 = (360) " + " ".join(to_str(yxx24_shifted[8])))
        str_list.append("ph10 = (360) " + " ".join(to_str(yxx24_shifted[9])))
        str_list.append("ph11 = (360) " + " ".join(to_str(yxx24_shifted[10])))
        str_list.append("ph12 = (360) " + " ".join(to_str(yxx24_shifted[11])))

        return f"\n".join(str_list)
