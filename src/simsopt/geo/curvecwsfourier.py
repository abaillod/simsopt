import numpy as np
import jax.numpy as jnp
from jax import vjp, jacfwd, jvp

from .jit import jit
import simsoptpp as sopp
from .._core.json import GSONDecoder
from .._core.derivative import Derivative
from .curve import Curve, JaxCurve, kappa_pure, torsion_pure
from .surfacerzfourier import SurfaceRZFourier
from .._core.optimizable import Optimizable

__all__ = ['CurveCWSFourier', 'CurveCWSFourier2']

def gamma_curve_on_surface(modes, qpts, order, surf_dofs, mpol, ntor, nfp):
    # Unpack dofs
    phic = modes[:order+1]
    phis = modes[order+1:2*order+1]
    thetac   = modes[2*order+1:3*order+2]
    thetas   = modes[3*order+2:]

    # Construct theta and phi arrays
    theta = jnp.zeros((qpts.size,))
    phi = jnp.zeros((qpts.size,))

    ll = qpts*2.0*jnp.pi
    for ii in range(order+1):
        theta = theta + thetac[ii] * jnp.cos(ii*ll)
        phi   = phi   + phic[ii]   * jnp.cos(ii*ll)

    for ii in range(order):
        theta = theta + thetas[ii] * jnp.sin((ii+1)*ll)
        phi   = phi   + phis[ii]   * jnp.sin((ii+1)*ll)


    # Construct curve on surface
    r = jnp.zeros((qpts.size,))
    z = jnp.zeros((qpts.size,))

    nmn = ntor+1 + mpol*(2*ntor+1)
    rc = surf_dofs[:nmn]
    zs = surf_dofs[nmn:]

    th = theta * 2.0 * jnp.pi
    ph = phi   * 2.0 * jnp.pi * nfp
    
    #ph = (1.0+phi) * jnp.pi / (2.0*nfp)

    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<0:
                continue
            counter = counter+1
            r = r + rc[counter] * jnp.cos(mm*th - nn*ph)


    counter = -1
    for mm in range(mpol+1):
        for nn in range(-ntor,ntor+1):
            if mm==0 and nn<=0:
                continue
            counter = counter+1
            z = z + zs[counter] * jnp.sin(mm*th - nn*ph)
            
    gamma = jnp.zeros((qpts.size, 3))
    gamma = gamma.at[:,0].set( r * jnp.cos( phi*jnp.pi*2 ) )
    gamma = gamma.at[:,1].set( r * jnp.sin( phi*jnp.pi*2 ) )
    gamma = gamma.at[:,2].set( z                 )

    return gamma

class CurveCWSFourier( JaxCurve ):
    def __init__(self, quadpoints, order, surf:SurfaceRZFourier, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = jnp.linspace(0, 1, quadpoints, endpoint=False)
        
        # Curve order. Number of Fourier harmonics for phi and theta
        self.order = order

        # Set surface
        if not surf.stellsym:
            raise ValueError('Not implemented for non-stellarator symmetric surfaces')

        if not isinstance(surf, SurfaceRZFourier):
            raise ValueError('This implementation assumes the surface to be a SurfaceRZFourier object')
        self.surf = surf
        
        # Modes are order as phic, phis, thetac, thetas
        self.modes = [np.zeros((order+1,)), np.zeros((order,)), np.zeros((order+1,)), np.zeros((order,))]

        # Define pure function
        pure = jit(lambda dofs, points: gamma_curve_on_surface(dofs, points, self.order, self.surf.x, self.surf.mpol, self.surf.ntor, self.surf.nfp))


        # Call JaxCurve constructor
        if dofs is None:
            super().__init__(
                quadpoints, 
                pure, 
                x0=np.concatenate(self.modes),
                external_dof_setter=CurveCWSFourier.set_dofs_impl,
                names=self._make_names()
                )
        else:
            super().__init__(
                quadpoints, 
                pure, 
                dofs=dofs,
                external_dof_setter=CurveCWSFourier.set_dofs_impl,
                names=self._make_names()
                )

    def num_dofs(self):
        return 2*(self.order+1) + 2*self.order
    
    def get_dofs(self):
        return np.concatenate(self.modes)

    def set_dofs_impl(self, dofs):
        self.modes[0] = dofs[0:self.order+1]
        self.modes[1] = dofs[self.order+1:2*self.order+1]
        self.modes[2] = dofs[2*self.order+1:3*self.order+2]
        self.modes[3] = dofs[3*self.order+2:4*self.order+2]

    def _make_names(self):
        dofs_name = []
        for mode in ['phic', 'phis', 'thetac', 'thetas']:
            for ii in range(self.order+1):
                if mode=='phis' and ii==0:
                    continue

                if mode=='thetas' and ii==0:
                    continue

                dofs_name.append(f'{mode}({ii})')

        return dofs_name
