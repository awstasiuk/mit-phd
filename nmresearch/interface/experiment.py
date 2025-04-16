import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nmrglue as ng
import numpy as np
from numpy import (
    imag,
    real,
    argmax,
    meshgrid,
    array,
    argmin,
    arange,
    polyfit,
    log,
    exp,
    pi,
    pad,
)
from scipy.interpolate import CubicSpline
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks


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

    def __init__(self, expt_no, folder="expt8", data_path=None):
        self.data_path = (
            "C:\\Users\\awsta\\Dropbox\\NMR_Data" if data_path is None else data_path
        )
        self.file = self.data_path + "\\" + folder + "\\" + str(expt_no)
        self.nmr_dic, self.nmr_data = ng.fileio.bruker.read(self.file)
        self.td = self.nmr_data.shape

    def calibrate90(self, use_imag=True):
        if len(self.td) != 3:
            raise ValueError("Invalid Experiment shape for pulse calibration")

        vals = (
            imag(self.nmr_data[:, :, 0]) if use_imag else real(self.nmr_data[:, :, 0])
        )
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

        print("array index of max signal is " + str(argmax(high_contrast)))

    def calibrate_framechange(self, phase_inc=1, offset=0, use_real=True):
        signal = (
            real(self.nmr_data[:, :, 0]) if use_real else imag(self.nmr_data[:, :, 0])
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y = meshgrid(
            range(offset, phase_inc * (signal.shape[0]) + offset, phase_inc),
            range(signal.shape[1]),
        )
        c = ax.plot_wireframe(x, y, signal.transpose())
        ax.set_title("Out of Phase Over-rotation Error")
        # set the limits of the plot to the limits of the data
        ax.set_xlabel("Frame Change Offset")
        ax.set_ylabel("Variable Delay List Element")
        plt.show()

        norms = [sum(abs(signal[k, :])) for k in range(self.td[0])]
        phases = array(range(offset, phase_inc * (signal.shape[0]) + offset, phase_inc))
        plt.scatter(phases, norms)
        cs = CubicSpline(phases, norms)
        xs = arange(phases[0], phases[-1], phase_inc / 50)
        plt.plot(xs, cs(xs), "g", label="Cubic Spline")

        plt.xlabel("Frame Change Offset")
        plt.title("1-norm of Pulse Error")
        plt.ylabel("(Absolute) Deviation from Ideal")
        plt.show()

        overrot_angle = phase_inc * (argmin(norms))
        print("Over-rotation angle is given by " + str(overrot_angle) + " deg.")

    def fid(self, add_spline=True, normalize=False, title="FID"):
        vals_real = real(self.nmr_data)
        vals_imag = imag(self.nmr_data)

        if normalize:
            vals_real = vals_real / vals_real[0]
            vals_imag = vals_imag / vals_real[0]

        t_real = [20 + 2 * i * 3.350 for i in range(len(vals_real))]
        t_imag = [20 + (2 * i + 1) * 3.350 for i in range(len(vals_imag))]

        if add_spline:
            xs = arange(t_real[0], t_imag[-1], 1)

            cs_real = CubicSpline(t_real, vals_real)
            plt.plot(xs, cs_real(xs), color="b", label="Real Spline")

            cs_imag = CubicSpline(t_imag, vals_imag)
            plt.plot(xs, cs_imag(xs), color="r", label="Imaginary Spline")

        plt.scatter(
            t_real, vals_real, marker="x", label="Real Component", s=15, color="b"
        )
        plt.scatter(
            t_imag, vals_imag, marker="x", label="Imaginary Component", s=15, color="r"
        )

        plt.xlabel("Evolution time (us)")
        plt.ylabel("Signal Intensity")
        plt.title(title)
        plt.legend()
        plt.show()

        return vals_real, vals_imag

    def offset_cal(self, use_spline=False, pad_factor=2, width=5e4):
        signal = pad(self.nmr_data, (0, (2**pad_factor - 1) * len(self.nmr_data)))

        t_real = [2 * i * 3.350 for i in range(len(signal))]
        t_imag = [(2 * i + 1) * 3.350 for i in range(len(signal))]

        t0 = t_real[0]
        tf = t_imag[-1]

        if use_spline:
            dt = 10**-2

            xs = arange(t0, tf, dt)
            cs_real = CubicSpline(t_real, real(signal))
            cs_imag = CubicSpline(t_imag, imag(signal))

            smooth_signal = cs_real(xs) + 1j * cs_imag(xs)

            ft = fftshift(fft(smooth_signal, norm="ortho"))
            N = len(ft)
            freq = fftshift(fftfreq(N, dt))
        else:
            dt = 3.350
            ft = fftshift(fft(signal, norm="ortho"))
            N = len(ft)
            freq = fftshift(fftfreq(N, dt))

        plt.plot(freq * 10**6, abs(ft) / N, label="Fourier Intensity")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Fourier Transform")
        plt.title("Offset Calibration")
        plt.xlim([-width, width])

        arr, _ = find_peaks(abs(ft) / N, height=max(abs(ft) / N) / 2)
        idx = arr[(len(arr) // 2) - 1]
        center = freq[idx] * 10**6

        print("Offset Frequency is approximately " + str(center) + "Hz")
        print("Use with caution, fine tuning may be necessary.")

        plt.plot([center], [abs(ft[idx]) / N], marker="o", markersize=6)
        plt.legend()
        plt.show()

        return freq[arr]

    def tpc(
        self,
        use_real=True,
        title="TPC Experiment",
        dipolar=False,
        add_spline=True,
        normalize=True,
        cycle=120,
    ):
        t_list = list(range(0, self.td[0] * cycle, cycle))
        if not dipolar:
            vals = real(self.nmr_data[:, 0]) if use_real else imag(self.nmr_data[:, 0])
        else:
            vals = imag(self.nmr_data[:, 1])

        if normalize:
            vals = vals / vals[0]

        if add_spline:
            cs = CubicSpline(t_list, vals)
            xs = arange(t_list[0], t_list[-1], cycle / 100)
            plt.plot(xs, cs(xs), "g", label="Cubic Spline")
        plt.scatter(t_list, vals, marker="o", label="data")
        plt.xlabel("Evolution time (us)")
        plt.ylabel("(Absolute) Signal Intensity")
        plt.title(title)
        plt.legend()
        plt.show()

        return vals

    def load_tpc(self, use_real=True, dipolar=False, normalize=True, error_bar=False):
        if not dipolar:
            vals = real(self.nmr_data[:, 0]) if use_real else imag(self.nmr_data[:, 0])
        else:
            vals = imag(self.nmr_data[:, 1])

        errs = np.std(np.real(self.nmr_data)[:,-50:-1],axis=1)

        if normalize:
            vals = vals / vals[0]
            errs = errs / vals[0]
        if error_bar:
            return  vals,errs
        return vals

    def load_tpc3d(self, use_real=True, dipolar=False, normalize=True):
        if not dipolar:
            vals = (
                real(self.nmr_data[:, :, 0])
                if use_real
                else imag(self.nmr_data[:, :, 0])
            )
        else:
            vals = imag(self.nmr_data[:, :, 1])

        # normalize signal
        if normalize:
            for idx, expt in enumerate(vals):
                vals[idx] = expt / max(expt)
        return vals

    def mqc(self, enc_td2=True, cycle=24 * 5, use_real=True):
        if use_real:
            Smt = real(self.nmr_data[:, :, 0])
        else:
            Smt = imag(self.nmr_data[:, :, 0])

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
        p = polyfit(t_list, log(fidelity), 1)
        tau_half = -log(2) / p[0]
        plt.scatter(t_list, fidelity, marker="x", label="Data")
        plt.plot(
            t_list,
            [exp(p[0] * t + p[1]) for t in t_list],
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

        Iqt = array(
            [
                (1 / (2 * M))
                * sum(
                    [exp(1j * q * idx * (pi / M)) * val for idx, val in enumerate(mat)]
                )
                for q in range(M + 1)
            ]
        )

        mqc = abs(Iqt)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y = meshgrid(range(mqc.shape[0]), range(mqc.shape[1]))
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

        vals = [abs(val / norm) for val, norm in zip(otoc, normalizer)]
        cs = CubicSpline(t_list, vals)
        xs = arange(t_list[0], t_list[-1], 1)
        plt.scatter(
            t_list,
            vals,
            label="data",
        )
        plt.plot(xs, cs(xs), "g", label="Cubic Spline")
        plt.xlabel("Evolution Time (us)")
        plt.ylabel("(Normalized) Signal Intensity")
        plt.title("<|[O(t),P]|^2> = Sum(q^2 * I_q(t)) / Fidelity(t)")
        plt.legend()
        plt.show()

        return mqc
