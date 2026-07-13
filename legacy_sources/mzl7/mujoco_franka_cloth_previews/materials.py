"""Cloth material presets and derived continuum stiffness metadata.

The cloth is a MuJoCo flexcomp grid (dim=2 shell) with:
  - stretch resistance from edge equality constraints (near-inextensible),
  - bending resistance from the ``mujoco.elasticity.shell`` plugin
    (Young's modulus ``young``, Poisson ratio ``poisson``, ``thickness``).

The stiffness numbers reported in metadata are the classic thin-shell
continuum values derived from (young, poisson, thickness):

  stretch_stiffness (membrane) : E*t / (1 - nu^2)        [N/m]
  bend_stiffness    (flexural) : E*t^3 / (12*(1 - nu^2)) [N*m]
  shear_stiffness   (membrane) : E*t / (2*(1 + nu))      [N/m]

Note: MuJoCo's edge-equality stretch is effectively stiffer than the
membrane value below; shear is not independently controllable for a grid
flex. These are nominal/derived values for preview metadata, not fitted
physical parameters.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MaterialPreset:
    label: str
    young: float          # Pa, shell plugin Young's modulus (drives bending)
    poisson: float        # shell plugin Poisson ratio
    thickness: float      # m, shell thickness
    mass: float           # kg, total cloth mass
    edge_damping: float   # flex edge damping
    fric_table: float     # cloth-table sliding friction
    fric_grip: float      # cloth-gripper sliding friction
    rgba: tuple = field(default=(0.8, 0.3, 0.3, 1.0))

    @property
    def stretch_stiffness(self) -> float:
        return self.young * self.thickness / (1.0 - self.poisson**2)

    @property
    def bend_stiffness(self) -> float:
        return self.young * self.thickness**3 / (12.0 * (1.0 - self.poisson**2))

    @property
    def shear_stiffness(self) -> float:
        return self.young * self.thickness / (2.0 * (1.0 + self.poisson))

    def density(self, area: float) -> float:
        """Volumetric density [kg/m^3] given cloth area [m^2]."""
        return self.mass / (area * self.thickness)

    def areal_density(self, area: float) -> float:
        """Areal density [kg/m^2] given cloth area [m^2]."""
        return self.mass / area


# Physics-variant presets. Variant v uses PRESETS[v % len(PRESETS)] exactly
# (no per-variant jitter) so that counterfactual bundles share identical
# scripted actions and differ only in physics.
PRESETS = [
    MaterialPreset("cotton_medium", young=3.0e3, poisson=0.3, thickness=8.0e-4,
                   mass=0.10, edge_damping=0.010, fric_table=0.70, fric_grip=1.00,
                   rgba=(0.85, 0.35, 0.30, 1.0)),
    MaterialPreset("silk_soft", young=8.0e2, poisson=0.3, thickness=5.0e-4,
                   mass=0.06, edge_damping=0.005, fric_table=0.40, fric_grip=0.80,
                   rgba=(0.35, 0.55, 0.85, 1.0)),
    MaterialPreset("denim_stiff", young=2.0e4, poisson=0.3, thickness=1.2e-3,
                   mass=0.18, edge_damping=0.020, fric_table=0.90, fric_grip=1.20,
                   rgba=(0.25, 0.30, 0.55, 1.0)),
    MaterialPreset("knit_stretchy", young=1.5e3, poisson=0.3, thickness=1.0e-3,
                   mass=0.12, edge_damping=0.008, fric_table=0.60, fric_grip=0.90,
                   rgba=(0.45, 0.75, 0.40, 1.0)),
    MaterialPreset("canvas_heavy", young=5.0e4, poisson=0.3, thickness=1.5e-3,
                   mass=0.24, edge_damping=0.030, fric_table=1.00, fric_grip=1.10,
                   rgba=(0.80, 0.70, 0.45, 1.0)),
]
