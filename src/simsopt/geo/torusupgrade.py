import jax.numpy as jnp
from math import pi, sin, cos
import numpy as np
from .curve import JaxCurve
from simsopt._core.optimizable import Optimizable

__all__ = ['HBTCylFourier','create_equally_spaced_cylindrical_curves']


def xyz_cyl(minor_r, l):
    """
    given curve dofs return a curve centered at the origin
    """
    nl = l.size
    out = jnp.zeros((nl,3))
    out = out.at[:,0].set( minor_r*jnp.cos(2.*np.pi*l))
    out = out.at[:,1].set(0.0) 
    out= out.at[:,2].set( minor_r*jnp.sin(2.*np.pi*l))
    return out

def rotations(curve,minor_r, alpha_r,alpha_phi,alpha_z,dr):
    #rotates curves around r,phi,z coordinates

    z_rot = jnp.asarray(
        [[jnp.cos(alpha_z), -jnp.sin(alpha_z), 0],
        [jnp.sin(alpha_z), jnp.cos(alpha_z), 0],
        [0, 0, 1]])

    y_rot = jnp.asarray(
        [[jnp.cos(alpha_phi), 0, jnp.sin(alpha_phi)],
        [0, 1, 0],
        [-jnp.sin(alpha_phi), 0, jnp.cos(alpha_phi)]])
    
    x_rot = jnp.asarray(
        [[1, 0, 0],
        [0, jnp.cos(alpha_r), -jnp.sin(alpha_r)],
        [0, jnp.sin(alpha_r), jnp.cos(alpha_r)]])

    out = curve

    out = out.at[:,2].set( out[:,2] + minor_r ) 
    out = out @ y_rot @x_rot @ z_rot 

    out = out.at[:,0].set( out[:,0] + dr ) 
    out = out.at[:,2].set( out[:,2] - minor_r ) 

    return out

def convert_to_cyl(a):
    #convert to cylindrical
    out = jnp.zeros(a.shape)
    out = out.at[:,0].set( jnp.sqrt(a[:,0]**2 + a[:,1]**2))
    out =out.at[:,1].set( jnp.arctan2(a[:,1],a[:,0]))
    out = out.at[:,2].set(a[:,2])

    return out

def cylindrical_shift(a,dphi,dz):
    #shifting in r
    out = jnp.zeros(a.shape)
    out = out.at[:,0].set(a[:,0])
    out = out.at[:,1].set(a[:,1]+dphi)
    out = out.at[:,2].set(a[:,2]+dz)
    return out

def cyl_to_cart(a):
    #cylindrical to cartesian
    out = jnp.zeros(a.shape)

    out = out.at[:,0].set(a[:,0] * jnp.cos(a[:,1]))
    out =out.at[:,1].set(a[:,0] * jnp.sin(a[:,1]))
    out = out.at[:,2].set(a[:,2])
    return out




def gamma_pure(dofs,points,minor_r):
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    g1= xyz_cyl(minor_r,points)
    g2 = rotations(g1,minor_r, ypr[0],ypr[1],ypr[2],xyz[0])
    g3 = convert_to_cyl(g2)
    g4 = cylindrical_shift(g3, xyz[1],xyz[2])
    final_gamma = cyl_to_cart(g4)
    return final_gamma


class HBTCylFourier(JaxCurve):
    """
    OrientedCurveCylindricalFourier is a translated and rotated Curve in r, theta, phi, and z coordinates.
    """

    def __init__(self, quadpoints, minor_r , dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        
        pure = lambda dofs, points: gamma_pure(dofs, points, minor_r)

        self.coefficients = [np.zeros((3,)), np.zeros((3,))]
        self.minor_r = minor_r
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=HBTCylFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=HBTCylFourier.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 3

    def get_dofs(self):
        return np.concatenate([self.coefficients[0], self.coefficients[1]])


    def set_dofs_impl(self, dofs):
        """
        This function sets the degrees of freedom (DoFs) for this object.
         """
        self.coefficients[0][:] = dofs[0:3]  # R0, phi, Z0
        self.coefficients[1][:] = dofs[3:6]  # theta, constant_phi,



    def _make_names(self):
        """
        Generates names for the degrees of freedom (DoFs) for this object.
        """

        rtpz_name = ['R0', 'phi', 'Z0']
        angle_name = ['r_rotation', 'phi_rotation', 'z_rotation']

        return rtpz_name + angle_name 

    @classmethod
    def convert_xyz_to_cyl(cls, curve_xyz_fourier, quadpoints=None,minor_r = None):
        """
        Converts a CurveXYZFourier object to an OrientedCurveCylindricalFourier object.

        Parameters:
        curve_xyz_fourier: Instance of CurveXYZFourier
        quadpoints: Quadrature points for the curve
        rotation_center: Center of rotation in cylindrical coordinates (r, phi, z)

        Returns:
        oriented_curve: Instance of OrientedCurveCylindricalFourier
        """
        if quadpoints is None:
            quadpoints = curve_xyz_fourier.quadpoints
        print(f"quadpoints: {quadpoints}")
        print(f"minor_r: {minor_r}")

       

        curve_dofs = curve_xyz_fourier.get_dofs()
        print(f"Original curve DoFs: {curve_dofs}")
        print(f"Sliced DoFs for cylindrical curve: {curve_dofs[:6]}")

        cylindrical_curve = cls(quadpoints, minor_r=minor_r)
        print(f"Cylindrical curve created with minor_r={minor_r}")

        cylindrical_curve.set_dofs(curve_dofs[:6])

      
      

        return cylindrical_curve

def create_equally_spaced_cylindrical_curves(ncurves, nfp, stellsym, R0=1.0, minor_r=0.5, numquadpoints=None):
   
    
    curves = []
    for i in range(ncurves):
        curve = HBTCylFourier(numquadpoints, minor_r)
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ncurves)
        curve.set('R0' ,R0)
        curve.set('phi' ,angle)
        curve.set('Z0' ,0)
        curve.set('r_rotation' ,0)
        curve.set('phi_rotation' ,0)
        curve.set('z_rotation' ,0)

        curves.append(curve)
    return curves
