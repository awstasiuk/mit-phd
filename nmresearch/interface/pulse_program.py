from numpy import array


def to_str(lst):
    return [str(val) for val in lst]


def reshape_pp(prog, target_rows):
    ph_prog_len = len(prog)
    ph_prog_depth = len(prog[0])
    n_pulse = ph_prog_depth * ph_prog_len
    if n_pulse % target_rows != 0:
        print("Incompatible shapes")
        return prog
    temp = array(prog)
    return temp.transpose().reshape((n_pulse // target_rows, target_rows)).transpose()


class TwoPointCorrelator:
    r"""
    A modular class for generating bruker pulse sequences for complex NMR Two
    Point Correlator experiments. There are a few pre-defined pulse sequences
    available as gloabl variables own by this class and its instances.

    """

    WHH = [[0], [270], [90], [180]]
    WHH8 = [
        [0, 180],
        [90, 270],
        [270, 90],
        [180, 0],
    ]
    WEI16 = [
        [0, 0, 180, 180],
        [90, 90, 270, 270],
        [90, 90, 270, 270],
        [0, 0, 180, 180],
    ]
    STABERYXX = [
        [90, 0, 90],
        [0, 0, 180],
        [0, 180, 180],
        [270, 180, 270],
    ]
    PENG24 = [
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
    PINE8 = [[0, 0, 180, 180], [0, 0, 180, 180]]  # t/2 - 2t - t - 2t - t ...
    MREV = [
        [180, 0],
        [90, 90],
        [270, 270],
        [0, 180],
    ]

    def __init__(self, use_magic=True):
        """
        Create an instance of this class, setting the measurement pulses and dictating
        if the measurement is to be a direct FID measurement (`use_magic`=False) or assisted
        with the solid echo (`use_magic`=True), which is the default behavior. 32 phase programs
        can be defined, but the last three (29,30,31) are reserved for measurement. Phase program
        labels are defined on-the-fly via a first-come first-serve procedure. That is, if the
        observable is set before the state, then the observable-related sequences will have lower
        phase program indices than the the state-related sequences. This should not have any
        adverse effect on the pulse prorgam's ability to run.
        """
        self.phase_programs = [None for _ in range(32)]
        if use_magic:
            self.phase_programs[-3] = [0, 180]
        self.phase_programs[-2] = [90, 270]
        self.phase_programs[-1] = [0, 2]
        # ph31 must be in base-4 for our topspin language

        self.idx = 0
        self.delay_defs = []

        self.state_is_set = False
        self.prep_loop = None
        self.prep_pattern = None

        self.obs_is_set = False
        self.obs_loop = None
        self.obs_pattern = None

        self.evo_is_set = False
        self.evo_range = None
        self.evo_pattern = None

    def set_prep_pattern(self, pattern, definitions):
        """
        this should check stuff and save things
        """
        if self.prep_loop is None:
            print("No observable sequence delays to specify")
            return
        prep_pulses = (self.prep_loop[1] - self.prep_loop[0] + 1) * len(
            self.phase_programs[self.prep_loop[0]]
        )
        if prep_pulses % (len(pattern) - 1) != 0:
            print("invalid delay pattern specified")
            return
        for delay in definitions:
            if delay not in self.delay_defs:
                self.delay_defs.append(delay)
        self.prep_pattern = pattern

    def set_evo_pattern(self, pattern, definitions):
        """
        this should check stuff and save things
        """
        if self.evo_range is None:
            print("Specify evolution sequence first")
            return

        for delay in definitions:
            if delay not in self.delay_defs:
                self.delay_defs.append(delay)
        self.evo_pattern = pattern

    def set_obs_pattern(self, pattern, definitions):
        """
        this should check stuff and save things
        """
        if self.obs_loop is None:
            print("No observable sequence delays to specify")
            return
        for delay in definitions:
            if delay not in self.delay_defs:
                self.delay_defs.append(delay)
        self.obs_pattern = pattern

    def set_global_state(self, state="Z"):
        """
        Set the initial state, global options only
        """
        if not self.state_is_set:
            match state:
                case "Z":
                    pass  # this is the default initial state
                case "X":
                    self.phase_programs[self.idx] = [90, 270]
                    self.idx += 1
                case "Y":
                    self.phase_programs[self.idx] = [180, 0]
                    self.idx += 1
                case "Dz":
                    print("Not Implemented :(")
                    return
                case "Dx":
                    print("Not Implemented :(")
                    return
                case "Dy":
                    print("Not Implemented :(")
                    return
                case _:
                    print("Command not recognized")
                    return
            self.state_is_set = True
        else:
            print("State is already defined")

    def set_disorder_state(self, state="rZ", seq=None, cycles=5):
        """
        Sets the initial state to be one of the disordered states
        """
        if not self.state_is_set:
            match state:
                case "rZ":
                    prog = seq if seq is not None else self.WHH8
                    self.phase_programs[self.idx] = [90, 90, 270, 270, 0, 0, 180, 180]
                    self.idx += 1
                    self.prep_loop = (self.idx, self.idx + len(prog) - 1)
                    for row in prog:
                        self.phase_programs[self.idx] = list(row) * cycles
                        self.idx += 1
                    self.phase_programs[self.idx] = [0, 0, 180, 180, 270, 270, 90, 90]
                    self.idx += 1
                case "rX":
                    pass
                case "rY":
                    pass

                case "rDz":
                    print("Not Implemented :(")
                    return
                case "rDx":
                    print("Not Implemented :(")
                    return
                case "rDy":
                    print("Not Implemented :(")
                    return

                case _:
                    print("Command not recognized")
                    return

            self.state_is_set = True
        else:
            print("State is already defined")

    def set_evolution(self, pulses, num_programs=None):
        """
        set the evolution phase program for the interior loop
        """
        if not self.evo_is_set:
            rows = len(pulses) if num_programs is None else num_programs
            prog = reshape_pp(pulses, rows)
            for offset, row in enumerate(prog):
                self.phase_programs[self.idx + offset] = row

            self.evo_range = (self.idx, self.idx + len(prog) - 1)
            self.idx += len(prog)
            self.evo_is_set = True
        else:
            print("inner evolution loop has already been defined")

    def set_global_observable(self, obs="Z"):
        """
        Set the observable, global options only
        """
        if not self.obs_is_set:
            match obs:
                case "Z":
                    pass  # this is the default observable
                case "X":
                    self.phase_programs[self.idx] = [270, 90]
                    self.idx += 1
                case "Y":
                    self.phase_programs[self.idx] = [0, 180]
                    self.idx += 1
                case "Dz":
                    print("Not Implemented :(")
                    return
                case "Dx":
                    print("Not Implemented :(")
                    return
                case "Dy":
                    print("Not Implemented :(")
                    return
                case _:
                    print("Command not recognized")
                    return
            self.obs_is_set = True
        else:
            print("Observable is already defined")

    def set_disorder_observable(self, obs="rZ", seq=None, cycles=5):
        """
        Sets the observable to be one of the disordered operators
        """
        if not self.obs_is_set:
            match obs:
                case "rZ":
                    prog = seq if seq is not None else self.WHH8
                    self.phase_programs[self.idx] = [0] * 8 + [180] * 8
                    self.idx += 1
                    self.obs_loop = (self.idx, self.idx + len(prog) - 1)
                    for row in prog:
                        self.phase_programs[self.idx] = list(row) * cycles
                        self.idx += 1
                    self.phase_programs[self.idx] = [270] * 8 + [90] * 8
                    self.idx += 1

                case "rX":
                    pass
                case "rY":
                    pass

                case "rDz":
                    print("Not Implemented :(")
                    return
                case "rDx":
                    print("Not Implemented :(")
                    return
                case "rDy":
                    print("Not Implemented :(")
                    return

                case _:
                    print("Command not recognized")
                    return

            self.obs_is_set = True
        else:
            print("Observable is already defined")

    def generate_phase_programs(self, fc=0, evo_max=1, print_me=True):
        """
        Generate the phase programs in the bruker format, printing the generated
        lists of strings in a nice way

        `fc` is the frame change angle to correct for phase transient errors

        `evo_max` is the amount of floquet periods of the evolution phase program to
        pre-compile. If `fc`!=0, then td2 of the TPC experiment should not exceed this
        quantity
        """
        if not (self.obs_is_set and self.state_is_set and self.evo_is_set):
            return "experiment is not yet fully defined"

        pp_list = []
        update_list = []
        reset_list = []
        glb_phase = 0

        # do the state prep pulses
        if self.evo_range[0] > 0:
            if self.prep_loop is not None:
                for idx in range(self.prep_loop[0]):
                    arr = array(self.phase_programs[idx])
                    pp_list.append(
                        f"ph{idx} = (360) " + " ".join(to_str((arr + glb_phase) % 360))
                    )
                    glb_phase += fc

                pp_list.append("")

                rows = self.prep_loop[1] - self.prep_loop[0] + 1
                depth = len(self.phase_programs[self.prep_loop[0]])
                shifts = array(
                    [
                        [fc * (rows * i + k) + glb_phase for i in range(depth)]
                        for k in range(rows)
                    ]
                )
                prog = array(
                    self.phase_programs[self.prep_loop[0] : self.prep_loop[1] + 1]
                )
                shifted_prog = shifts + prog
                for idx, phases in enumerate(shifted_prog):
                    pp_list.append(
                        f"ph{idx + self.prep_loop[0]} = (360) "
                        + " ".join(to_str(phases % 360))
                    )
                    reset_list.append(f"rpp{idx+self.prep_loop[0]}")
                glb_phase += fc * rows * depth

                pp_list.append("")

                for idx in range(self.prep_loop[1] + 1, self.evo_range[0]):
                    arr = array(self.phase_programs[idx])
                    pp_list.append(
                        f"ph{idx} = (360) " + " ".join(to_str((arr + glb_phase) % 360))
                    )
                    glb_phase += fc

            else:
                for idx in range(self.evo_range[0]):
                    arr = array(self.phase_programs[idx])
                    pp_list.append(
                        f"ph{idx} = (360) " + " ".join(to_str((arr + glb_phase) % 360))
                    )
                    glb_phase += fc

            pp_list.append("")

        # do the evolution stuff
        prog = array(self.phase_programs[self.evo_range[0] : self.evo_range[1] + 1])
        ph_prog_len = len(prog)
        ph_prog_depth = len(prog[0])
        n_pulses = ph_prog_depth * ph_prog_len
        evo_ph_delta = (n_pulses * fc) % 360
        shifts = [
            [
                glb_phase + fc * (ph_prog_len * i + k)
                for i in range(evo_max * ph_prog_depth)
            ]
            for k in range(ph_prog_len)
        ]
        prog_shifted = [
            [
                (prog[k][idx % ph_prog_depth] + shift) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]
        for idx, phases in enumerate(prog_shifted):
            pp_list.append(
                f"ph{idx + self.evo_range[0]} = (360) " + " ".join(to_str(phases))
            )
            reset_list.append(f"rpp{idx+self.evo_range[0]}")

        pp_list.append("")

        # do the observable prep pulses
        if self.evo_range[1] + 1 < self.idx:
            if self.obs_loop is not None:
                for idx in range(self.evo_range[1] + 1, self.obs_loop[0]):
                    arr = array(self.phase_programs[idx])
                    pp_list.append(
                        f"ph{idx} = (360) " + " ".join(to_str((arr + glb_phase) % 360))
                    )
                    update_list.append(f"25m ip{idx}*{evo_ph_delta}")
                    glb_phase += fc

                pp_list.append("")

                rows = self.obs_loop[1] - self.obs_loop[0] + 1
                depth = len(self.phase_programs[self.obs_loop[0]])
                shifts = array(
                    [
                        [fc * (rows * i + k) + glb_phase for i in range(depth)]
                        for k in range(rows)
                    ]
                )
                prog = array(
                    self.phase_programs[self.obs_loop[0] : self.obs_loop[1] + 1]
                )
                shifted_prog = shifts + prog
                for idx, phases in enumerate(shifted_prog):
                    pp_list.append(
                        f"ph{idx + self.obs_loop[0]} = (360) "
                        + " ".join(to_str(phases % 360))
                    )
                    update_list.append(f"25m ip{idx + self.obs_loop[0]}*{evo_ph_delta}")
                    reset_list.append(f"rpp{idx+self.obs_loop[0]}")
                glb_phase += fc * rows * depth

                pp_list.append("")

                for idx in range(self.obs_loop[1] + 1, self.idx):
                    arr = array(self.phase_programs[idx])
                    pp_list.append(
                        f"ph{idx} = (360) " + " ".join(to_str((arr + glb_phase) % 360))
                    )
                    update_list.append(f"25m ip{idx}*{evo_ph_delta}")
                    glb_phase += fc

            else:
                for idx in range(self.evo_range[1] + 1, self.idx):
                    arr = array(self.phase_programs[idx])
                    pp_list.append(
                        f"ph{idx} = (360) " + " ".join(to_str((arr + glb_phase) % 360))
                    )
                    update_list.append(f"25m ip{idx}*{evo_ph_delta}")
                    glb_phase += fc

            pp_list.append("")

        # do the readout sequence
        if self.phase_programs[-3] is not None:
            pp_list.append(f"ph29 = (360) " + " ".join(to_str(self.phase_programs[-3])))

        pp_list.append(f"ph30 = (360) " + " ".join(to_str(self.phase_programs[-2])))
        pp_list.append(f"ph31 = " + " ".join(to_str(self.phase_programs[-1])))
        if print_me:
            print("d1 " + " ".join(reset_list))
            print("")
            print(f"\n".join(update_list))
            print("")
            print(f"\n".join(pp_list))
        else:
            return reset_list, update_list, pp_list

    def generate_pulse_program(self, fc=0, evo_max=1, filename=None):
        """
        Generates a full bruker pulse program for a two point correlator experiment,
        as described by this class. The entire pulse program is printed as a string and
        saved to a textfile in the same directory as the active instance calling this
        method. the filename defaults to "default_pp.txt" if no filename is specified.

        `fc` is the frame change angle to correct for phase transient errors

        `evo_max` is the amount of floquet periods of the evolution phase program to
        pre-compile. If `fc`!=0, then td2 of the TPC experiment should not exceed this
        quantity
        """
        reset_list, update_list, pp_list = self.generate_phase_programs(
            fc, evo_max, print_me=False
        )
        #####
        # Header stuff
        #####
        header_list = []
        header_list.append("; delay definitions")
        for ddef in self.delay_defs:
            delay = ddef.split("=")[0]
            header_list.append(f"define delay {delay}")
            header_list.append('"' + ddef + '"')
        header_list.append('"l1=0"  ; start the evolution counter at 0')
        header_list.append("")
        header_list.append("1   ze")
        header_list.append("")
        header_list.append("2")
        header_list.append("d1 " + " ".join(reset_list))
        header_list.append("100u pl2:f2")
        header_list.append("")

        #####
        # State Logic
        #####
        state_list = []

        if self.evo_range[0] > 0:
            if self.prep_loop is not None:
                for idx in range(self.prep_loop[0]):
                    state_list.append(f"(p1 ph{idx}):f2")
                    state_list.append("1.5u")

                state_list.append("")

                prog = array(
                    self.phase_programs[self.prep_loop[0] : self.prep_loop[1] + 1]
                )
                ph_prog_len = len(prog)
                ph_prog_depth = len(prog[0])
                loops_per_prep = (ph_prog_depth * ph_prog_len) // (
                    len(self.prep_pattern) - 1
                )
                state_list.append("3")
                state_list.append(self.prep_pattern[0])
                for idx, delay in enumerate(self.prep_pattern[1:]):
                    state_list.append(f"(p1 ph{(idx+self.prep_loop[0])}^):f2")
                    state_list.append(delay)

                state_list.append("")

                state_list.append(f"lo to 3 times {loops_per_prep}")

                state_list.append("")

                for idx in range(self.prep_loop[1] + 1, self.evo_range[0]):
                    state_list.append(f"(p1 ph{idx}):f2")
                    state_list.append("1.5u")

                state_list.append("")
            else:
                for idx in range(self.evo_range[0]):
                    state_list.append(f"(p1 ph{idx}):f2")
                    state_list.append("1.5u")
                state_list.append("")

        #####
        # Evolution Logic
        #####
        evo_list = []

        prog = array(self.phase_programs[self.evo_range[0] : self.evo_range[1] + 1])
        ph_prog_len = len(prog)
        ph_prog_depth = len(prog[0])
        loops_per_pp = (ph_prog_depth * ph_prog_len) // (len(self.evo_pattern) - 1)

        evo_list.append("4")
        evo_list.append(self.evo_pattern[0])
        for idx, delay in enumerate(self.evo_pattern[1:]):
            evo_list.append(f"(p1 ph{self.evo_range[0] + (idx % ph_prog_len)}^):f2")
            evo_list.append(delay)

        evo_list.append("")
        evo_list.append("lo to 4 times l1")
        evo_list.append("")

        #####
        # Observable Logic
        #####
        obs_list = []

        if self.evo_range[1] + 1 < self.idx:
            if self.obs_loop is not None:
                for idx in range(self.evo_range[1] + 1, self.obs_loop[0]):
                    obs_list.append(f"(p1 ph{idx}):f2")
                    obs_list.append("1.5u")

                obs_list.append("")

                prog = array(
                    self.phase_programs[self.obs_loop[0] : self.obs_loop[1] + 1]
                )
                ph_prog_len = len(prog)
                ph_prog_depth = len(prog[0])
                loops_per_obs = (ph_prog_depth * ph_prog_len) // (
                    len(self.obs_pattern) - 1
                )
                obs_list.append("5")
                obs_list.append(self.obs_pattern[0])
                for idx, delay in enumerate(self.obs_pattern[1:]):
                    obs_list.append(f"(p1 ph{(idx+self.obs_loop[0])}^):f2")
                    obs_list.append(delay)

                obs_list.append("")

                obs_list.append(f"lo to 5 times {loops_per_obs}")

                obs_list.append("")

                for idx in range(self.obs_loop[1] + 1, self.idx):
                    obs_list.append(f"(p1 ph{idx}):f2")
                    obs_list.append("1.5u")

                obs_list.append("")
            else:
                for idx in range(self.evo_range[1] + 1, self.idx):
                    obs_list.append(f"(p1 ph{idx}):f2")
                    obs_list.append("1.5u")
                obs_list.append("")

        #####
        # Measurement and exit
        #####
        meas_list = []
        meas_list.append("1m")
        if self.phase_programs[-3] is not None:
            meas_list.append("(p1 ph29):f2")
            meas_list.append("20u")
        meas_list.append("(p1 ph30):f2")
        meas_list.append("")
        meas_list.append("go=2 ph31")
        meas_list.append("1m wr #0 if #0")
        meas_list.append("")
        # incremenent the evolution loop...
        for _ in range(loops_per_pp):
            meas_list.append("iu1")
        meas_list.append("")
        meas_list.extend(update_list)
        meas_list.append("")
        meas_list.append("lo to 2 times td1")
        meas_list.append("")
        meas_list.append("exit")
        meas_list.append("")

        print(f"\n".join(header_list))
        print(f"\n".join(state_list))
        print(f"\n".join(evo_list))
        print(f"\n".join(obs_list))
        print(f"\n".join(meas_list))
        print(f"\n".join(pp_list))

        file = filename if filename is not None else "default_pp.txt"
        with open(file, "w") as f:
            f.write(f"\n".join(header_list))
            f.write(f"\n".join(state_list))
            f.write(f"\n".join(evo_list))
            f.write(f"\n".join(obs_list))
            f.write(f"\n".join(meas_list))
            f.write(f"\n".join(pp_list))
