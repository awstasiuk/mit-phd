import nmrglue as ng


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
        self.file = self.data_path + "\\" + str(121)
        self.nmr_dic, self.nmr_data = ng.fileio.bruker.read(file)
