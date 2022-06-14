import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nmrglue as ng
import numpy as np


class Experiment:
    r"""
    A class which reads in Bruker NMR data and bundles a bunch of experimental analysis
    and optimization tools commonly used to perform various tasks on the spectrometer,
    such as

        - reading in arbitrary experimental data for pythonic analysis
        - tunning up Pi/2 pulse power
        - tunning up out-of-phase over-rotation error
        - reading in Two Point Correlator Experiments (option with Dipolar states)
        - reading in Multiple Quantum Coherence Experiments (option with Dipolar states)
        - Computing OTOC from MQC data results
    """

    def __init__(self, expt_no, data_path=None):
        self.data_path = (
            "C:\\Users\\awsta\\Dropbox\\NMR_Data" if data_path is None else data_path
        )
        self.file = self.data_path + "\\" + str(expt_no)
        self.nmr_dic, self.nmr_data = ng.fileio.bruker.read(self.file)
        self.td = self.nmr_data.shape

    def calibrate90(self):
        if len(self.td) != 3:
            raise ValueError("Invalid Experiment shape for pulse calibration")
        vals = np.imag(self.nmr_data[:, :, 0])
        plt.plot(vals[0], label="1 wrap")
        plt.plot(vals[1], label="2 wraps")
        plt.plot(vals[2], label="3 wraps")
        plt.title("Bloch Sphere Wrapping Signal")
        plt.xlabel("Power List Index")
        plt.legend()
        plt.show()

        high_contrast = vals[0] * vals[1] * vals[2]
        plt.plot(high_contrast)
        plt.title("High contrast signal plot")
        plt.xlabel("Power List Index")
        plt.show()

        print("array index of max signal is " + str(np.argmax(high_contrast)))

    def calibrate_framechange(self, phase_inc=2):
        signal = np.real(self.nmr_data[:, :, 0])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y = np.meshgrid(
            range(0, phase_inc * (signal.shape[0]), phase_inc),
            range(signal.shape[1]),
        )
        c = ax.plot_wireframe(x, y, signal.transpose())
        ax.set_title("Out of Phase Over-rotation Error")
        # set the limits of the plot to the limits of the data
        ax.set_xlabel("Frame Change Offset")
        ax.set_ylabel("Variable Delay List Element")
        plt.show()

        norms = [sum(abs(signal[k, :])) for k in range(self.td[0])]
        plt.plot(list(range(0, phase_inc * (signal.shape[0]), phase_inc)), norms)
        plt.xlabel("Frame Change Offset")
        plt.title("1-norm of Pulse Error")
        plt.ylabel("(Absolute) Deviation from Ideal")
        plt.show()

        overrot_angle = phase_inc * (np.argmin(norms))
        print("Over-rotation angle is given by " + str(overrot_angle) + " deg.")

    def mqc(self, enc_td2=True):
        cycle = 24 * 5.0
        Smt = np.real(self.nmr_data[:, :, 0])

        if enc_td2:
            td1 = self.td[0]
            td2 = self.td[1]
            fidelity = Smt[:, 0]
            mat = Smt.transpose()
        else:
            td1 = self.td[1]
            td2 = self.td[0]
            fidelity = Smt[0, :]
            mat = Smt

        t_list = [n * cycle for n in range(td1)]
        M = int(td2 / 2)
        p = np.polyfit(t_list, np.log(fidelity), 1)
        tau_half = -np.log(2) / p[0]
        plt.scatter(t_list, fidelity, marker="x", label="Data")
        plt.plot(
            t_list,
            [np.exp(p[0] * t + p[1]) for t in t_list],
            "g",
            label="Exponential Fit",
        )
        plt.legend()
        plt.xlabel("Evolution Length (us)")
        plt.ylabel("Time Reversal Signal Intensity")
        plt.title("< (O') O >, O' = V(-t)*U(-t)*O*U(t)*V(t), V(t)~U(-t)")
        plt.show()
        print(
            "Fidelity decays with half-life of "
            + str(tau_half)
            + "us, or "
            + str(tau_half / cycle)
            + " cycles."
        )

        Iqt = np.array(
            [
                (1 / (2 * M))
                * sum(
                    [
                        np.exp(1j * q * idx * (np.pi / M)) * val
                        for idx, val in enumerate(mat)
                    ]
                )
                for q in range(M + 1)
            ]
        )

        mqc = np.abs(Iqt)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y = np.meshgrid(range(mqc.shape[0]), range(mqc.shape[1]))
        c = ax.plot_wireframe(x, y, mqc.transpose())
        ax.set_title("MQC with M=" + str(M) + " encoding cycles")
        # set the limits of the plot to the limits of the data
        ax.set_xlabel("Coherence order index")
        ax.set_ylabel("Evolution cycles")
        plt.show()

        plt.plot(t_list, mqc[0, :], label="q=0")
        plt.plot(t_list, mqc[2, :], label="q=2")
        plt.plot(t_list, mqc[4, :], label="q=4")
        plt.xlabel("Evolution Time (us)")
        plt.ylabel("(Absolute) Signal Intensity")
        plt.legend()
        plt.title("Even order coherence evolution")
        plt.show()

        plt.plot(t_list, mqc[1, :], label="q=1")
        plt.plot(t_list, mqc[3, :], label="q=3")
        plt.plot(t_list, mqc[5, :], label="q=5")
        plt.xlabel("Evolution Time (us)")
        plt.ylabel("(Absolute) Signal Intensity")
        plt.legend()
        plt.title("Odd order coherence evolution")
        plt.show()

        otoc = sum([(q**2) * intensity for q, intensity in enumerate(Iqt)])
        normalizer = fidelity
        plt.plot(
            t_list,
            [8 * np.abs(val / norm) for val, norm in zip(otoc, normalizer)],
            label="Full MQC Summation",
        )
        plt.xlabel("Evolution Time (us)")
        plt.ylabel("(Normalized) Signal Intensity")
        plt.title("<|[O(t),P]|^2> = Sum(q^2 * I_q(t)) / Fidelity(t)")
        plt.show()

        return mqc
