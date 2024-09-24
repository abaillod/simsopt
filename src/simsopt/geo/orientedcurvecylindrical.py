import jax.numpy as jnp
from math import pi, sin, cos
import numpy as np
from .curve import JaxCurve
from simsopt._core.optimizable import Optimizable

__all__ = ['OrientedCurveXYZFourierCyl']

def shift_pure( v, xyz ):
    xyz = cylindrical_to_cartesian(xyz_cyl)
    for ii in range(0,3):
        v = v.at[:,ii].add(xyz[ii])
    return v

#Shifts a set of vectors by a specified amount in 3D space

def rotate_pure( v, ypr ):        
    yaw = ypr[0]
    pitch = ypr[1]
    roll = ypr[2]

    Myaw = jnp.asarray(
        [[jnp.cos(yaw), -jnp.sin(yaw), 0],
        [jnp.sin(yaw), jnp.cos(yaw), 0],
        [0, 0, 1]]
    )
    Mpitch = jnp.asarray(
        [[jnp.cos(pitch), 0, jnp.sin(pitch)],
        [0, 1, 0],
        [-jnp.sin(pitch), 0, jnp.cos(pitch)]]
    )
    Mroll = jnp.asarray(
        [[1, 0, 0],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll), jnp.cos(roll)]]
    )

    return v @ Myaw @ Mpitch @ Mroll

def cartesian_to_cylindrical(v):
    """Converts Cartesian coordinates to cylindrical coordinates."""
    x, y, z = v
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    return jnp.array([r, theta, z])

def cylindrical_to_cartesian(v):
    """Converts cylindrical coordinates to Cartesian coordinates."""
    r, theta, z = v
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.array([x,y,z])

def centercurve_pure(dofs, quadpoints, order, rotation_center=None):
    """
    Performs rotations and shifts on a curve represented by Fourier series coefficients in cylindrical coordinates.

    Args:
        dofs (np.ndarray): Array of DOFs, including translation in cylindrical coordinates, rotation angles, and Fourier series coefficients.
        quadpoints (np.ndarray): Array of quadrature points for the curve.
        order (int): Order of the Fourier series.
        rotation_center (np.ndarray, optional): Rotation center in Cartesian coordinates. Defaults to None.

    Returns:
        np.ndarray: Array of final curve points in Cartesian coordinates.
    """
    rotation_center = cylindrical_to_cartesian(rotation_center_cyl) if rotation_center_cyl is not None else jnp.zeros(3)

    # Extract components from DOFs
    xyz_cyl = dofs[0:3]  # Translation in cylindrical coordinates (radial, angular, vertical)
    ypr = dofs[3:6]  # Rotation angles (yaw, pitch, roll)
    fmn = dofs[6:]  # Fourier series coefficients

    # Convert cylindrical translation to Cartesian for rotation center calculation
    rotation_center_cartesian = cylindrical_to_cartesian(xyz_cyl) if rotation_center is not None else None

    # Convert quadrature points to cylindrical coordinates
    quadpoints_cyl = cartesian_to_cylindrical(quadpoints)

    # Generate Fourier series terms in cylindrical coordinates (radial and angular components)
    gamma_cyl_r = jnp.zeros((len(quadpoints_cyl),))
    gamma_cyl_theta = jnp.zeros((len(quadpoints_cyl),))
    for i in range(order):
        gamma_cyl_r += fmn[2 * i] * jnp.sin(2 * pi * (i + 1) * quadpoints_cyl[:, 1])
        gamma_cyl_r += fmn[2 * i + 1] * jnp.cos(2 * pi * (i + 1) * quadpoints_cyl[:, 1])
        gamma_cyl_theta += fmn[2 * i + 2 * order] * jnp.sin(2 * pi * (i + 1) * quadpoints_cyl[:, 1])
        gamma_cyl_theta += fmn[2 * i + 2 * order + 1] * jnp.cos(2 * pi * (i + 1) * quadpoints_cyl[:, 1])

    # Combine radial and angular components into cylindrical coordinates
    gamma_cyl = jnp.stack([gamma_cyl_r, gamma_cyl_theta, quadpoints_cyl[:, 2]], axis=1)

    # Shift the curve in cylindrical coordinates
    shifted_gamma_cyl = shift_pure(gamma_cyl, xyz_cyl - rotation_center_cyl)

    # Apply rotations in cylindrical coordinates (around z-axis, then around x-axis, then around y-axis)
    rotated_gamma_cyl = rotate_pure(shifted_gamma_cyl, ypr)

    # Shift the rotated curve back to the original position
    final_gamma_cyl = shift_pure(rotated_gamma_cyl, rotation_center_cyl + xyz_cyl)

    # Convert final curve points back to Cartesian coordinates
    final_gamma = cylindrical_to_cartesian(final_gamma_cyl)

    return final_gamma

    #gives a point as input which can be gotten through the gamma function, and then the output should be the curve rotated around 
    #this point and not zero
    #this is a rotate then shift
    #in order to position all the curves at zero, may need to shift, rotate, then shift 

    #need to modify the class so that one can rotate around any point, then constraint so that the point can be set at any point in
    #cartesian space

    #have to define how to rotate around the first gamma point, then make a penalty function where it has to curl around that point

class OrientedCurveXYZFourierCyl( JaxCurve ):
    """
    OrientedCurveXYZFourierCyl is a translated and rotated 
    JaxCurveXYZFourier Curve.
    """
    def __init__(self, quadpoints, order, dofs=None, rotation_center_cyl = np.zeros(3)):
        rotation_center_cyl = cylindrical_to_cartesian(rotation_center_cyl)
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        self.rotation_center_cyl=rotation_center_cyl
        self.order = order
        pure = lambda dofs, points: centercurve_pure(dofs, points, self.order, rotation_center_cyl)
        #we go from dof to real space coordinate here, by modifying centercurvepure.
        #need to to provide an addition attribute that the user can set which is the point, which needs to be passed to the 
        #centercurve pure function which also being passed in as gamma
        self.coefficients = [np.zeros((3,)), np.zeros((3,)), np.zeros((2*order,)), np.zeros((2*order,)), np.zeros((2*order,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=OrientedCurveXYZFourierCyl.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=OrientedCurveXYZFourierCyl.set_dofs_impl,
                             names=self._make_names())
                    
            self.update_rotation_center()


    #def update_rotation_center(self, x0):
        #"""Updates the rotation center based on the current curve"""
        #self.rotation_center = x0
        #curve_points = self.gamma()
        #if len(curve_points) > 0:
            #self.rotation_center = curve_points[0]
        #else:
        #    raise ValueError("Curve points are empty, cannot set rotation center")

#I think the issue may be that this isn't being assigned to the class, so that the first value is always chosen as 
#the point of rotation


    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 3 + 3*(2*self.order)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)
    
    def set_dofs_impl(self, dofs):
        self.coefficients[0][:] = dofs[0:3]
        self.coefficients[1][:] = dofs[3:6]

        counter = 6
        for i in range(0,3):
            for j in range(0, self.order):
                self.coefficients[i+2][2*j] = dofs[counter]
                counter += 1
                self.coefficients[i+2][2*j+1] = dofs[counter]
                counter += 1

        

    def _make_names(self):
        xyc_name = ['x0', 'y0', 'z0']
        ypr_name = ['yaw', 'pitch', 'roll']
        dofs_name = []
        for c in ['x', 'y', 'z']:
            for j in range(0, self.order):
                dofs_name += [f'{c}s({j+1})', f'{c}c({j+1})']
        return xyc_name + ypr_name + dofs_name

    @classmethod
    def convert_xyz_to_oriented(cls, curve_xyz_fourier, quadpoints=None,rotation_center_cyl=None):
        """
        Converts a CurveXYZFourier object to an OrientedCurveXYZFourier object.

        Parameters:
        curve_xyz_fourier: Instance of CurveXYZFourier
        quadpoints: Quadrature points for the curve
        order: Fourier series order

        Returns:
        oriented_curve: Instance of OrientedCurveXYZFourier
        """

        if quadpoints is None:
            quadpoints = curve_xyz_fourier.quadpoints
        
        fixed_point = curve_xyz_fourier.gamma()[0]

        curve_dofs = curve_xyz_fourier.get_dofs()
        o = curve_xyz_fourier.order
        translation = curve_dofs[[0, 2*o+1,2*(2*o+1)]]
        curve_dofs = np.delete(curve_dofs,[0, 2*o+1,2*(2*o+1)])

        # Initialize translation and rotation to zero
        
        rotation = np.zeros(3)

        # Combine the translation, rotation, and curve DOFs
        oriented_dofs = np.concatenate([translation, rotation, curve_dofs])

        oriented_curve = cls(quadpoints, curve_xyz_fourier.order, rotation_center_cyl=rotation_center_cyl)

        oriented_curve.set_dofs(oriented_dofs)
        #have to loop through all the dofs of oriented curve and give them the value i want them to hold, which has a
        #1to1 correspondence with xyzfourier dof

        #c.set('xo',...)

        return oriented_curve



#make shift and rotation in cylindrical coordinate system