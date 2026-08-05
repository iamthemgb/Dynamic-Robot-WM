"""Synthetic text prompts, one per appearance bucket.

f1_10h carries no language instructions -- ``meta/tasks.parquet`` has no task
string column and nothing else in the corpus supplies one. Wan's cross
attention still needs a context tensor, so prompts are synthesised from the
taxonomy: same key as ``f1_groups.GROUP_KEYS``, hence ~32-40 distinct strings.

That the prompts are near-identical across episodes is the point, not a
shortcut: the text channel carries essentially no discriminative signal, which
forces the physics adapters to do the conditioning work. It is also why the
metric to trust is the PAIRED gap (correct vs wrong records under shared
sigma/eps) rather than raw flow loss.
"""

SUBFAMILY = {
    "centered_vertical_drop": "falling straight down onto its fingers",
    "off_center_drop": "falling off to one side of its fingers",
    "drifted_drop": "drifting sideways as it falls toward its fingers",
    "mild_projectile": "arcing toward it on a shallow projectile trajectory",
}
TOOL = {
    "franka_hand": "a Franka Panda robot arm with a two-finger parallel gripper",
    "robotiq_2f85_thick_pad": "a Franka Panda robot arm with a Robotiq 2F-85 "
                              "gripper with thick pads",
}
BACKGROUND = {
    "clean_franka_lab": "in a clean, empty laboratory",
    "robocasa_lab": "in a cluttered laboratory",
    "robocasa_kitchen": "in a kitchen",
    "robocasa_workbench": "at a workbench",
    "robocasa_tabletop": "on a tabletop scene",
}
VARIANT = {
    "catch_retain": "catches and holds the ball",
    "catch_transport": "catches the ball and carries it",
    "off_center_catch": "reaches out and catches the ball",
    "off_center_near_miss": "reaches for the ball",
    "drift_catch": "tracks and catches the ball",
    "drift_near_miss": "tracks the ball",
    "mild_projectile_catch": "intercepts and catches the ball",
    "mild_projectile_near_miss": "reaches toward the ball",
}


#: robot-free passive corpora (subfamily -> full prompt); the arm/gripper
#: template below is meaningless for them.
ROBOTLESS = {
    "rolling_dynamics": "A small red ball rolls across a kitchen island "
                        "countertop, in a kitchen. Static camera, side view.",
}

#: single-fixed-prompt corpora (subfamily -> the corpus's one prompt). rbi
#: reuses the exact string of the zl664 runs so latents+text match; the
#: primary path repacks the cached embedding (rbi_convert_cache) and this
#: entry only serves the t5_cache fallback.
FIXED = {
    "rolling_intercept": "A Franka robot intercepts and physically grasps "
                         "a ball rolling across a kitchen table.",
}


def bucket_key(row):
    return (str(row.leaf), str(row.subfamily), str(row.variant),
            str(row.tool_type), str(row.background_style))


def prompt_for(leaf, subfamily, variant, tool_type, background_style) -> str:
    if subfamily in FIXED:
        return FIXED[subfamily]
    if subfamily in ROBOTLESS:
        return ROBOTLESS[subfamily]
    tool = TOOL.get(tool_type, "a robot arm with a parallel-jaw gripper")
    action = VARIANT.get(variant, "reaches for the ball")
    motion = SUBFAMILY.get(subfamily, "falling toward its fingers")
    scene = BACKGROUND.get(background_style, "in a laboratory")
    return (f"{tool.capitalize()} {action}. A small ball is {motion}, "
            f"{scene}. Static camera, three-quarter view.")


def prompt_table(idx):
    """-> {prompt_id: text} for an index carrying a prompt_id column."""
    table = {}
    for r in idx.itertuples():
        table.setdefault(int(r.prompt_id), prompt_for(*bucket_key(r)))
    return dict(sorted(table.items()))
