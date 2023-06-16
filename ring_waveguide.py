import torch
from photontorch import photontorch as pt
import numpy as np


class Waveguide(pt.Component):
    """ Waveguide

    Each waveguide has two ports. They are numbered 0 and 1:

    Ports:

        0 ---- 1

    """

    num_ports = 2

    def __init__(
            self,
            length=1e-5,
            loss=0,  # in dB/m
            neff=2.34,  # effective index of the waveguide
            ng=3.40,  # group index of the waveguide
            wl0=1.55e-6,  # center wavelength for which the waveguide is defined
            phase=0,  # additional phase PARAMETER added to the waveguide
            trainable=True,  # a flag to make the component trainable or not
            name=None,  # name of the waveguide
    ):
        """ creation of a new waveguide """
        super(Waveguide, self).__init__(name=name)  # always initialize parent first
        # Handle inputs
        self.loss = float(loss)
        self.neff = float(neff)
        self.wl0 = float(wl0)
        self.ng = float(ng)
        self.length = float(length)

        # handle phase input
        phase = float(phase) % (2 * np.pi)
        if not trainable:  # if the network is not trainable, just store it as a normal float:
            self.phase = phase
        else:  # else, make an optimizable parameter out of it:
            # create a torch tensor from the phase
            phase = torch.tensor(phase, dtype=torch.float64)
            # store the phase as an optimizable parameter
            self.phase = torch.nn.Parameter(data=phase)

    def set_delays(self, delays):
        """ set the delays for time-domain simulations """
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):

        current_simulation_environment = self.env

        # you can use this environment to get information about the
        # wavelengths used in the simulation:
        wavelength = current_simulation_environment.wavelength

        wavelength = torch.tensor(
            wavelength,  # make this numpy array into a torch tensor
            dtype=torch.float64,  # keep float64 dtype
            device=self.device,  # put it on the current device ('cpu' or 'gpu')
        )

        # next we implement the dispersion, which will depend on the
        # wavelength tensor
        neff = self.neff - (wavelength - self.wl0) * (self.ng - self.neff) / self.wl0

        # we have now calculated an neff for each different wavelength.
        # let's calculate the phase depending on this neff:
        phase = (2 * np.pi * neff * self.length / wavelength) % (2 * np.pi)

        # next, we add the phase correction parameter.
        phase = phase + self.phase

        cos_phase = torch.cos(phase).to(torch.get_default_dtype())
        sin_phase = torch.sin(phase).to(torch.get_default_dtype())

        # finally, we can calculate the loss and add it to the phase, which
        # gives us the S-matrix parameters
        loss = 10 ** (-self.loss * self.length / 20)  # 20 because loss works on power
        re = loss * cos_phase
        ie = loss * sin_phase

        # the last thing to do is to add the S-matrix parameters to the S-matrix:
        S[0, :, 0, 1] = S[0, :, 1, 0] = re
        S[1, :, 0, 1] = S[1, :, 1, 0] = ie


class DirectionalCoupler(pt.Component):
    r""" A directional coupler is a component with 4 ports that introduces no delays

    Each directional coupler has four ports. They are numbered 0 to 3:

    Ports:
       3        2
        \______/
        /------\
       0        1

    """

    num_ports = 4

    def __init__(self, coupling=0.5, name=None):
        """ creation of a new waveguide """
        super(DirectionalCoupler, self).__init__(name=name)  # always initialize parent first

        parameter = torch.tensor(np.arccos(float(coupling)), dtype=torch.get_default_dtype())
        self.parameter = torch.nn.Parameter(data=parameter)

    @property
    def coupling(self):
        return torch.cos(self.parameter)

    def set_S(self, S):
        t = (1 - self.coupling) ** 0.5
        k = self.coupling ** 0.5

        # real part scattering matrix (transmission):
        S[0, :, 0, 1] = S[0, :, 1, 0] = t  # same for all wavelengths
        S[0, :, 2, 3] = S[0, :, 3, 2] = t  # same for all wavelengths

        # imag part scattering matrix (coupling):
        S[1, :, 0, 2] = S[1, :, 2, 0] = k  # same for all wavelengths
        S[1, :, 1, 3] = S[1, :, 3, 1] = k  # same for all wavelengths


class AllPass(pt.Network):
    def __init__(
            self,
            ring_length=1e-5,  # [um] length of the ring
            ring_loss=1,  # [dB]: roundtrip loss in the ring
            name=None
    ):
        super(AllPass, self).__init__(name=name)  # always initialize parent first

        # handle arguments:
        self.ring_length = ring_length
        self.ring_loss = ring_loss,

        # define subcomponents
        self.source = pt.Source()
        self.detector = pt.Detector()
        self.dc = DirectionalCoupler()
        self.wg = Waveguide(length=ring_length, loss=int(ring_loss / ring_length))
        self.link('source:0', '0:dc:2', '0:wg:1', '3:dc:1', '0:detector')


class AllPassRingResonator(pt.Network):
    def __init__(
            self,
            ring_length=1e-5,  # [um] length of the ring
            ring_loss=1,  # [dB]: roundtrip loss in the ring
            wg_length=1e-5,
            wg_loss=1,
            name=None
    ):
        super(AllPassRingResonator, self).__init__(name=name)  # always initialize parent first

        # handle arguments:
        self.ring_length = ring_length
        self.ring_loss = ring_loss,

        # define subcomponents
        self.source = pt.Source()
        self.detector = pt.Detector()
        self.dc = DirectionalCoupler()
        self.ring = Waveguide(length=ring_length, loss=int(ring_loss / ring_length))
        self.wg = Waveguide(length=wg_length, loss=int(wg_loss / wg_length))
        self.link('source:0', '0:dc:2', '0:ring:1', '3:dc:1', '0:wg:1', '0:detector')
