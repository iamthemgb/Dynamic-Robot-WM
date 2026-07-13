"""Rope material presets and honest stiffness metadata.

The rope is a MuJoCo composite ``type="cable"``: a chain of rigid capsule
bodies connected by ball joints, with bending/twist elasticity from the
``mujoco.elasticity.cable`` plugin (config keys ``twist``/``bend`` are shear
and Young's moduli in Pa).

Honesty notes (also emitted per-episode as `parameter_implementation_notes`):
  - stretch_stiffness is NOT adjustable: cable links are rigid, so the rope
    is inextensible by construction. The metadata field is null.
  - bend_stiffness is reported as flexural rigidity EI = E * pi * r^4 / 4
    [N*m^2] derived from the plugin Young's modulus and rope radius.
  - rope_density is the capsule geom volumetric density [kg/m^3].
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RopeMaterial:
    label: str
    length: float         # m
    radius: float         # m
    density: float        # kg/m^3 capsule geom density
    young_bend: float     # Pa, cable plugin "bend"
    shear_twist: float    # Pa, cable plugin "twist"
    joint_damping: float  # ball joint damping
    fric_table: float
    fric_grip: float
    fric_post: float
    fric_ring: float
    rgba: tuple

    @property
    def bend_stiffness_EI(self) -> float:
        """Flexural rigidity EI [N*m^2]."""
        return self.young_bend * np.pi * self.radius**4 / 4.0

    @property
    def twist_stiffness_GJ(self) -> float:
        """Torsional rigidity GJ [N*m^2]."""
        return self.shear_twist * np.pi * self.radius**4 / 2.0

    @property
    def linear_density(self) -> float:
        """kg/m."""
        return self.density * np.pi * self.radius**2


# Physics-variant presets. Variant v uses PRESETS[v % len(PRESETS)] exactly
# (no jitter), so counterfactual bundles share identical scripted actions and
# initial grasped-endpoint placement; rope length grows away from the grasped
# end, which stays fixed across variants.
PRESETS = [
    RopeMaterial("medium_rope", length=0.60, radius=0.007, density=800.0,
                 young_bend=8.0e6, shear_twist=8.0e5, joint_damping=0.01,
                 fric_table=0.70, fric_grip=1.00, fric_post=0.60, fric_ring=0.40,
                 rgba=(0.76, 0.55, 0.30, 1.0)),
    RopeMaterial("light_string", length=0.55, radius=0.004, density=500.0,
                 young_bend=1.5e6, shear_twist=2.0e5, joint_damping=0.004,
                 fric_table=0.50, fric_grip=0.90, fric_post=0.50, fric_ring=0.35,
                 rgba=(0.90, 0.88, 0.80, 1.0)),
    RopeMaterial("stiff_cable", length=0.65, radius=0.009, density=1300.0,
                 young_bend=8.0e7, shear_twist=2.0e7, joint_damping=0.03,
                 fric_table=0.85, fric_grip=1.10, fric_post=0.70, fric_ring=0.50,
                 rgba=(0.20, 0.22, 0.26, 1.0)),
    RopeMaterial("thick_hemp", length=0.62, radius=0.010, density=720.0,
                 young_bend=1.4e7, shear_twist=1.4e6, joint_damping=0.014,
                 fric_table=0.80, fric_grip=1.05, fric_post=0.65, fric_ring=0.45,
                 rgba=(0.62, 0.50, 0.32, 1.0)),
    RopeMaterial("nylon_cord", length=0.66, radius=0.006, density=1050.0,
                 young_bend=4.0e6, shear_twist=6.0e5, joint_damping=0.008,
                 fric_table=0.45, fric_grip=0.85, fric_post=0.42, fric_ring=0.30,
                 rgba=(0.15, 0.35, 0.70, 1.0)),
    RopeMaterial("red_paracord", length=0.58, radius=0.005, density=650.0,
                 young_bend=2.5e6, shear_twist=3.5e5, joint_damping=0.006,
                 fric_table=0.55, fric_grip=0.95, fric_post=0.50, fric_ring=0.35,
                 rgba=(0.72, 0.16, 0.18, 1.0)),
    RopeMaterial("thin_wire", length=0.52, radius=0.0035, density=2400.0,
                 young_bend=1.6e8, shear_twist=4.0e7, joint_damping=0.02,
                 fric_table=0.40, fric_grip=0.80, fric_post=0.45, fric_ring=0.35,
                 rgba=(0.55, 0.56, 0.60, 1.0)),
    RopeMaterial("green_garden_hose", length=0.64, radius=0.011, density=950.0,
                 young_bend=3.0e7, shear_twist=6.0e6, joint_damping=0.022,
                 fric_table=0.75, fric_grip=1.05, fric_post=0.62, fric_ring=0.48,
                 rgba=(0.18, 0.45, 0.28, 1.0)),
]

PARAMETER_IMPLEMENTATION_NOTES = {
    "stretch_stiffness": (
        "not implemented: composite cable links are rigid capsules connected "
        "by ball joints, so the rope is inextensible; field is null"),
    "bend_stiffness": (
        "cable plugin Young's modulus 'bend' [Pa]; reported bend_stiffness is "
        "the derived flexural rigidity EI = E*pi*r^4/4 [N*m^2]"),
    "twist_stiffness": (
        "cable plugin shear modulus 'twist' [Pa]; reported twist_stiffness is "
        "the derived torsional rigidity GJ = G*pi*r^4/2 [N*m^2]"),
    "rope_density": "capsule geom volumetric density [kg/m^3]",
    "frictions": (
        "rope-X sliding friction is the rope capsule geom friction; MuJoCo "
        "combines pair friction as elementwise max with the other geom"),
    "grasp": (
        "endpoint attachment is an equality connect (weld of endpoint body to "
        "hand frame), not a friction pinch; see endpoint_attachment_method"),
}
