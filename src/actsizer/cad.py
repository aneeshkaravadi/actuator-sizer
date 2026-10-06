"""Parametric rotary-actuator envelope: motor bay, gearbox bay, output flange, mounting ears.

This is a packaging study (does a motor of this diameter plus a gearbox fit at the
joint, and where do the bolts go?), not a detailed design. Dimensions in mm.
"""
from __future__ import annotations

from pathlib import Path


def actuator_housing(step_path: str | Path, motor_od_mm: float = 90.0, stack_mm: float = 35.0,
                     gearbox_mm: float = 40.0, wall_mm: float = 3.0, flange_bolts: int = 8):
    import build123d as bd

    od = motor_od_mm + 2 * wall_mm
    length = stack_mm + gearbox_mm + 2 * wall_mm
    with bd.BuildPart() as p:
        bd.Cylinder(od / 2, length, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        # motor bay (open at the back for the encoder cap) and gearbox bay
        with bd.Locations((0, 0, wall_mm)):
            bd.Cylinder(motor_od_mm / 2, stack_mm, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN),
                        mode=bd.Mode.SUBTRACT)
        with bd.Locations((0, 0, wall_mm + stack_mm)):
            bd.Cylinder(motor_od_mm / 2 - 4, gearbox_mm + wall_mm,
                        align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        # cable exit
        with bd.Locations((0, 0, 0)):
            bd.Cylinder(8, wall_mm, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        # mounting ears on the motor end
        for x in (-1, 1):
            with bd.Locations((x * (od / 2 + 6), 0, wall_mm)):
                bd.Box(16, 22, 2 * wall_mm + 4, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
            with bd.Locations((x * (od / 2 + 8), 0, 0)):
                bd.Cylinder(2.75, 2 * wall_mm + 6, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN),
                            mode=bd.Mode.SUBTRACT)
    # output flange as a separate ring with a bolt circle
    with bd.BuildPart() as fl, bd.Locations((0, 0, length)):
        bd.Cylinder(od / 2 - 6, 8, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        bd.Cylinder(12, 8, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        with bd.PolarLocations(od / 2 - 14, flange_bolts):
            bd.Cylinder(2.2, 8, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
    asm = bd.Compound(label="actuator_envelope", children=[p.part, fl.part])
    bd.export_step(asm, str(step_path))
    return asm
