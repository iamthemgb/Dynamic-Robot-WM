"""Soft-body material presets and honest implementation notes.

The soft object is a MuJoCo volumetric flex (``flexcomp type="grid" dim="3"``):
a tetrahedral FEM lattice with continuum elasticity from the built-in
``<elasticity young=... poisson=...>`` element.

Stability/contact findings on MuJoCo 3.3.1 (empirical, this cluster):
  - ``<elasticity damping>`` > 0 makes the flex blow up unconditionally
    (BADQACC warnings + silent auto-reset loops), even in free fall, with
    every integrator tried. We therefore set elasticity damping = 0 and
    implement damping as flex *edge* damping (per-edge dashpots).
  - Explicit elastic forces bound the stable timestep: dt must shrink as
    young/(1-2*poisson) grows. Each preset carries its own dt_sim
    (verified stable with a full press cycle on the ~11 mm grids). poisson
    0.45 at high young diverged at every dt tried, so presets cap at 0.40.
  - MuJoCo contact stiffness is inertia-scaled: with ~0.1 g flex vertices,
    the default solref (0.01 s) yields ~5 N/m per contact and any pusher
    tunnels straight through the flex ("hand goes through the object").
    The scene builder sets contact solref timeconst = 2*dt_sim (the
    documented stability floor), which is 100-400x stiffer here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SoftMaterial:
    label: str
    young: float          # Pa, flex <elasticity young>
    poisson: float        # flex <elasticity poisson>
    density: float        # kg/m^3 -> flexcomp mass = density * object volume
    edge_damping: float   # flex <edge damping>, the implemented damping
    damping_label: str    # low / medium / high (qualitative axis from spec)
    friction: float       # flex contact sliding friction
    dt_sim: float         # stability-limited timestep for this stiffness
    rgba: tuple

    def mass_for_volume(self, volume_m3: float) -> float:
        return self.density * volume_m3


# Physics-variant presets. Variant v uses PRESETS[v % len(PRESETS)] exactly
# (no jitter): counterfactual bundles share object geometry, initial pose and
# the scripted action; only these material parameters (and the
# stability-mandated dt_sim) change.
PRESETS = [
    SoftMaterial("soft_plush", young=2.0e4, poisson=0.30, density=120.0,
                 edge_damping=0.05, damping_label="low", friction=1.0,
                 dt_sim=2.5e-4, rgba=(0.87, 0.38, 0.55, 1.0)),
    SoftMaterial("medium_sponge", young=8.0e4, poisson=0.35, density=250.0,
                 edge_damping=0.15, damping_label="medium", friction=0.8,
                 dt_sim=2.0e-4, rgba=(0.93, 0.72, 0.18, 1.0)),
    SoftMaterial("stiff_foam", young=3.0e5, poisson=0.40, density=400.0,
                 edge_damping=0.35, damping_label="high", friction=0.5,
                 dt_sim=1.25e-4, rgba=(0.55, 0.65, 0.78, 1.0)),
]

PARAMETER_IMPLEMENTATION_NOTES = {
    "young_modulus": (
        "physically implemented: flexcomp <elasticity young> [Pa], continuum "
        "FEM elasticity on the tetrahedral grid"),
    "poisson_ratio": (
        "physically implemented: flexcomp <elasticity poisson>; capped at "
        "0.40 because 0.45 with high young diverges on MuJoCo 3.3.1 at every "
        "timestep tried"),
    "density": (
        "physically implemented as total flexcomp mass = density * nominal "
        "object volume, distributed uniformly over vertices (flexcomp has a "
        "mass attribute, not a density attribute)"),
    "damping": (
        "approximation: MuJoCo 3.3.1 built-in flex <elasticity damping> > 0 "
        "is unconditionally unstable (BADQACC auto-reset loops, reproduced "
        "even in free fall), so elasticity damping is 0 and dissipation is "
        "implemented as flex <edge damping> per-edge dashpots; the "
        "low/medium/high damping axis maps to edge damping 0.05/0.15/0.35, "
        "not to a calibrated Rayleigh coefficient"),
    "friction": (
        "physically implemented: flex contact sliding friction "
        "(torsional/rolling left at defaults); MuJoCo combines contact-pair "
        "friction as the elementwise max with the other geom"),
    "restitution": (
        "not implemented: the MuJoCo contact model has no restitution "
        "coefficient; effective bounce emerges from contact solref (0.01, 1); "
        "metadata field is null"),
    "plasticity": (
        "not implemented: flex elasticity is fully elastic (no plastic or "
        "viscoelastic terms available); field is false"),
    "mesh_resolution": (
        "flexcomp grid count per axis; fixed inside physics bundles, varied "
        "only in a separate mesh-resolution ablation if run"),
    "dt_sim": (
        "varies per material preset (2.5e-4 / 2e-4 / 1.25e-4 s) because "
        "explicit flex elastic forces bound the stable timestep; "
        "counterfactual bundles share the scripted action but not dt_sim"),
    "contact_stiffness": (
        "flex contact solref timeconst is set to 2*dt_sim (MuJoCo stability "
        "floor), not the physical material stiffness: MuJoCo contact "
        "stiffness is inertia-scaled and the ~0.1 g flex vertices make the "
        "default 0.01 s timeconst so soft that rigid bodies tunnel through "
        "the flex; this also means effective contact stiffness co-varies "
        "with dt_sim across physics variants"),
}
