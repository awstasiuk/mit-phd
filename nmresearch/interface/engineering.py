import pandas as pd
import numpy as np

I = np.eye(3)
x = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
x2 = np.matmul(x, x)
x3 = np.matmul(x2, x)
x = x2 @ x3
y = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
y2 = np.matmul(y, y)
y3 = np.transpose(y)
y = y2 @ y3
nmrTranslateS = {"0": x, "1": y, "2": x3, "3": y3}
nmrTranslate = {0: x, 1: y, 2: x3, 3: y3}


def to_str(lst):
    return [str(val) for val in lst]


class PulseProgram:

    def __init__(self, sequence, time):
        """store sequence information"""

        if type(sequence) == str:
            pulse = [I]
            for i in sequence:
                pulse.append(nmrTranslateS[i])
            self.s = pulse
        else:
            self.s = sequence
        self.t = time
        self.p = self.pulse_phase()
        pass

    def avg_ham(self, tP=0, Identity=1):
        """return 0th order Hamiltonian"""
        s = ""
        k = np.diag(self.k_vector(tP).T[0])
        L = np.diag(self.L_vector(tP).T[0])
        L = L - L[0][0] * np.eye(3) * Identity
        if k[0][0] != 0:
            s += f"{k[0][0]:.3f} rX +"
        if k[1][1] != 0:
            s += f" {k[1][1]:.3f} rY +"
        if k[2][2] != 0:
            s += f" {k[2][2]:.3f} rZ +"
        if L[0][0] != 0:
            s += f" {L[0][0]:.3f} Dx +"
        if L[1][1] != 0:
            s += f" {L[1][1]:.3f} Dy +"
        if L[2][2] != 0:
            s += f" {L[2][2]:.3f} Dz +"
        if s == "":
            s = "0 +"
        s = s[:-2]
        return s

    def error(self, tP, *, Identity=1):
        s = ""
        k = np.diag(self.k_vector().T[0])
        L = np.diag(self.L_vector().T[0])
        ktP = np.diag(self.k_vector(tP).T[0])
        LtP = np.diag(self.L_vector(tP).T[0])
        errorK = ktP - k
        errorL = LtP - L
        errorL = errorL - errorL[0][0] * np.eye(3) * Identity
        if errorK[0][0] != 0:
            s += f"{errorK[0][0]:.3f} rX +"
        if errorK[1][1] != 0:
            s += f" {errorK[1][1]:.3f} rY +"
        if errorK[2][2] != 0:
            s += f" {errorK[2][2]:.3f} rZ +"
        if errorL[0][0] != 0:
            s += f" {errorL[0][0]:.3f} Dx +"
        if errorL[1][1] != 0:
            s += f" {errorL[1][1]:.3f} Dy +"
        if errorL[2][2] != 0:
            s += f" {errorL[2][2]:.3f} Dz +"
        if s == "":
            s = "0 +"
        s = s[:-2]
        return s

    def f_matrix(self):
        """Return Frame Matrix from a Pulse Sequence."""
        FMat = [[], [], []]
        currentState = I
        for i in range(len(self.s)):
            currentState = self.s[i] @ currentState
            for j in range(3):
                FMat[j].append(currentState[2][j])
        return np.array(FMat)

    def k_vector(self, tP=0):
        """Return weighted row sum of Frame Matrix."""
        K = [[], [], []]
        for i in range(3):
            K[i].append(
                np.dot((self.f_matrix()[i]), np.array(self.t) + tP * 4 / np.pi)
                - self.f_matrix()[i][-1] * tP * 4 / np.pi
            )
        return 1 / (sum(self.t) + tP * (np.size(self.t) - 1)) * np.array(K)

    def L_vector(self, tP=0):
        """Return weighted absolute row sum of Frame Matrix."""
        L = [[], [], []]
        pulse = abs(self.f_matrix())
        for i in range(3):
            L[i].append(np.dot(pulse[i], np.array(self.t) + tP) - pulse[i][-1] * tP)
        return 1 / (sum(self.t) + tP * (np.size(self.t) - 1)) * np.array(L)

    def chirality_operator1(self):
        operatorX = []
        for k in range(len(np.transpose(self.f_matrix())) - 1):
            operatorX.append(
                np.cross(
                    (np.transpose(self.f_matrix())[k]),
                    (np.transpose(self.f_matrix())[k + 1]),
                )
            )
        return np.einsum("ji, i", np.transpose(operatorX), np.ones(len(operatorX)))

    def chirality_operator2(self):
        operatorX = []
        for k in range(len(np.transpose(self.f_matrix())) - 1):
            operatorX.append(
                np.cross(
                    (np.transpose(self.f_matrix())[k]),
                    (np.transpose(self.f_matrix())[k + 1]),
                )
            )
        return np.transpose(operatorX)

    def parity_operator1(self):
        P = np.zeros((3, np.size(self.f_matrix(), 1) - 1))
        for j in range(3):
            for k in range(np.size(P, 1)):
                P[j][k] = (
                    self.f_matrix()[j][k] * self.f_matrix()[(j + 1) % 3][k + 1]
                    + self.f_matrix()[(j + 1) % 3][k] * self.f_matrix()[j][k + 1]
                )
        return P @ np.ones(np.size(P, 1))

    def parity_operator2(self):
        P = np.zeros((3, np.size(self.f_matrix(), 1) - 1))
        for j in range(3):
            for k in range(np.size(P, 1)):
                P[j][k] = (
                    self.f_matrix()[j][k] * self.f_matrix()[(j + 1) % 3][k + 1]
                    + self.f_matrix()[(j + 1) % 3][k] * self.f_matrix()[j][k + 1]
                )
        return P

    def data_frame(self):
        a = pd.DataFrame(self.f_matrix().astype(int), index=("X", "Y", "Z"))
        return (
            a.style.format(precision=1)
            .bar(align="mid", color=["salmon", "lightgreen"])
            .set_properties(**{"width": "15px", "border": "1.5px solid black"})
        )

    def data_frame_parity(self):
        a = self.parity_operator2().astype(int).tolist()
        b = self.parity_operator1().astype(int)
        for i in range(3):
            a[i].append(b[i])
        c = [x for x in range(len(a[0]) - 1)]
        c.append("Sum")
        return pd.DataFrame(a, index=("XY", "YZ", "ZX"), columns=c)

    def data_frame_chirality(self):
        a = self.chirality_operator2().astype(int).tolist()
        b = self.chirality_operator1().astype(int)
        for i in range(3):
            a[i].append(b[i])
        c = [x for x in range(len(a[0]) - 1)]
        c.append("Sum")
        return pd.DataFrame(a, index=("X", "Y", "Z"), columns=c)

    def frame_matrix(self):
        a = self.f_matrix().astype(int).tolist()
        a.append(self.t)
        return np.array(a)

    def data_frame_matrix(self):
        """Return Frame Matrix and Tau Vector"""
        a = self.frame_matrix().tolist()
        return pd.DataFrame(a, index=("X", "Y", "Z", "Time"))

    def phase_program(self, base=4):
        pulse = []
        ph0 = []
        ph1 = []
        ph2 = []
        ph3 = []

        if base != 4:
            ph0.append(f"({(base//4)*4})")
            ph1.append(f"({(base//4)*4})")
            ph2.append(f"({(base//4)*4})")
            ph3.append(f"({(base//4)*4})")
        for i in self.s[1:]:
            for j in range(4):
                if np.array_equiv(i, nmrTranslate[j]):
                    pulse.append(j * (base // 4))
        if len(pulse) % 4:
            return pulse
        else:
            while len(pulse):
                ph0.append(pulse.pop(0))
                ph1.append(pulse.pop(0))
                ph2.append(pulse.pop(0))
                ph3.append(pulse.pop(0))

            phaseProgram0 = ""
            for i in ph0:
                phaseProgram0 += f"{i} "
            phaseProgram0 = phaseProgram0[:-1]

            phaseProgram1 = ""
            for i in ph1:
                phaseProgram1 += f"{i} "
            phaseProgram1 = phaseProgram1[:-1]

            phaseProgram2 = ""
            for i in ph2:
                phaseProgram2 += f"{i} "
            phaseProgram2 = phaseProgram2[:-1]

            phaseProgram3 = ""
            for i in ph3:
                phaseProgram3 += f"{i} "
            phaseProgram3 = phaseProgram3[:-1]
            phasePrograms = f"ph0 = {phaseProgram0}\nph1 = {phaseProgram1}\nph2 = {phaseProgram2}\nph3 = {phaseProgram3}"
            return phasePrograms

    def pulse_phase(self, base=360):
        pulse = []
        ph0 = []
        ph1 = []
        ph2 = []
        ph3 = []
        for i in self.s[1:]:
            for j in range(4):
                if np.array_equiv(i, nmrTranslate[j]):
                    pulse.append(j * (base // 4))
        if len(pulse) % 4:
            return pulse
        else:
            while len(pulse):
                ph0.append(pulse.pop(0))
                ph1.append(pulse.pop(0))
                ph2.append(pulse.pop(0))
                ph3.append(pulse.pop(0))
            pulse.append(ph0)
            pulse.append(ph1)
            pulse.append(ph2)
            pulse.append(ph3)
            return pulse

    def pi(self, fc, n_max, xx=False):

        ph_prog_len = len(self.p)
        ph_prog_depth = len(self.p[0])

        shifts = [
            [fc * (ph_prog_len * i + k) for i in range(n_max * ph_prog_depth)]
            for k in range(ph_prog_len)
        ]

        self_shifted = [
            [
                (self.p[k][idx % ph_prog_depth] + shift) % 360
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

        for idx, phases in enumerate(self_shifted):
            str_list.append("ph" + str(idx) + " = (360) " + " ".join(to_str(phases)))

        return f"\n".join(str_list)

    def disorder_state_internal_evolution(self, fc, n_prep=4):
        # preliminaries
        ph_prog_len = len(self.p)
        ph_prog_depth = len(self.p[0])
        n_pulses = ph_prog_depth * ph_prog_len

        # generate the frame change shifts for a single phase program of the right shape
        shifts = [
            [fc * (ph_prog_len * i + k) for i in range(n_prep * ph_prog_depth)]
            for k in range(ph_prog_len)
        ]

        glb_ph = 0

        # begin chaos

        # state prep
        ph0 = [90, 90, 270, 270, 0, 0, 180, 180]
        glb_ph += fc

        # ken num 1
        fwd_shifted = [
            [
                (self.p[k][idx % ph_prog_depth] + shift + glb_ph) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]
        glb_ph += n_pulses * n_prep * fc

        # cycle
        ph5 = [(ang + glb_ph) % 360 for ang in [0, 0, 180, 180, 270, 270, 90, 90]]
        glb_ph += fc

        #
        # evolution is a loop over delays
        #

        # observable stuff
        ph14 = [
            (ang + glb_ph) % 360
            for ang in [0, 0, 0, 0, 0, 0, 0, 0, 180, 180, 180, 180, 180, 180, 180, 180]
        ]
        glb_ph += fc

        # ken numb 2
        bwd_shifted = [
            [
                (self.p[k][idx % ph_prog_depth] + shift + glb_ph) % 360
                for idx, shift in enumerate(shifts[k])
            ]
            for k in range(ph_prog_len)
        ]
        glb_ph += n_pulses * n_prep * fc

        # recovery
        ph19 = [
            (ang + glb_ph) % 360
            for ang in [
                270,
                270,
                270,
                270,
                270,
                270,
                270,
                270,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
            ]
        ]

        str_list = []

        str_list.append("")
        str_list.append("ph0 = (360) " + " ".join(to_str(ph0)))
        str_list.append("")
        str_list.append("ph1 = (360) " + " ".join(to_str(fwd_shifted[0])))
        str_list.append("ph2 = (360) " + " ".join(to_str(fwd_shifted[1])))
        str_list.append("ph3 = (360) " + " ".join(to_str(fwd_shifted[2])))
        str_list.append("ph4 = (360) " + " ".join(to_str(fwd_shifted[3])))
        str_list.append("")
        str_list.append("ph5 = (360) " + " ".join(to_str(ph5)))
        str_list.append("")
        str_list.append("ph14 = (360) " + " ".join(to_str(ph14)))
        str_list.append("")
        str_list.append("ph15 = (360) " + " ".join(to_str(bwd_shifted[0])))
        str_list.append("ph16 = (360) " + " ".join(to_str(bwd_shifted[1])))
        str_list.append("ph17 = (360) " + " ".join(to_str(bwd_shifted[2])))
        str_list.append("ph18 = (360) " + " ".join(to_str(bwd_shifted[3])))
        str_list.append("")
        str_list.append("ph19 = (360) " + " ".join(to_str(ph19)))
        str_list.append("")

        return f"\n".join(str_list)

    def lab_z_in_tog(self):
        f = self.f_matrix().astype(int)
        test = []
        for i in range(len(self.s)):
            for j in range(3):
                if f[j][i]:
                    test.append((j + 1) * f[j][i])
        return test

    def symmetry_test(self):
        f = self.f_matrix().astype(int)
        test = []
        for i in range(len(self.s)):
            for j in range(3):
                if f[j][i]:
                    test.append((j + 1) * f[j][i])
        testr = test[::-1]
        if test == testr:
            return True
        return False

    def symmetry_test_rotations(self):
        pulse = self.s
        track = []
        for i in range(len(self.s) - 1):
            if PulseProgram(pulse, self.t).symmetry_test():
                track.append((i + len(self.s) // 2) % len(self.s))
            pulse.append(pulse.pop(1))
        if track != []:
            return track, True
        return False

    def suspension_sequence(self):
        suspulse = (np.array(self.p) // 90).T.flatten()
        suspulse = np.append(suspulse, suspulse)
        for i in range((len(suspulse) // 2) - 1, len(suspulse) - 1):
            suspulse[i] = ((suspulse[i]) + 2) % 4
        suspulse = suspulse.tolist()
        pulsetime = np.array(self.t)
        pulsetime[-1] = pulsetime[-1] + pulsetime[0]
        pulsetime = np.append(pulsetime, pulsetime[1:])
        pulsetime[-1] -= pulsetime[0]
        pulsetime = pulsetime.tolist()
        pulsestring = ""
        for i in suspulse:
            pulsestring += f"{i}"
        return PulseProgram(pulsestring, pulsetime)


WaHuHa = PulseProgram("0312", [1, 1, 2, 1, 1])
MREV8 = PulseProgram("21300132", [1, 1, 2, 1, 2, 1, 2, 1, 1])
WaHuHa8 = PulseProgram("21300312", [1, 1, 2, 1, 2, 1, 2, 1, 1])
opt12 = PulseProgram("030301030301", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

c1 = 1 / 4
u, v, w = 0, 0, 0
t1, T1 = 1 + c1 - v + w, 1 - c1 - v + w
t2, T2 = 1 - u + v, 1 - u + v
t3, T3 = 1 + u - w, 1 + u - w
wei16Time = [
    t1,
    t2,
    2 * t3,
    T2,
    2 * T1,
    t2,
    2 * T3,
    T2,
    2 * t1,
    T2,
    2 * T3,
    t2,
    2 * T1,
    T2,
    2 * t3,
    t2,
    t1,
]
wei16 = PulseProgram("0110011023322332", wei16Time)

STABER = PulseProgram("012321230103", [1 / 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 / 2])
STABER2 = PulseProgram("012101101210", STABER.t)
STABERYXX = PulseProgram("100300221223", STABER.t)
