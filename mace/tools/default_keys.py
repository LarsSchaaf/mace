from __future__ import annotations

from enum import Enum


class DefaultKeys(Enum):
    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    POLARIZABILITY = "polarizability"
    HEAD = "head"
    CHARGES = "REF_charges"
    TOTAL_CHARGE = "total_charge"
    TOTAL_SPIN = "total_spin"
    ELEC_TEMP = "elec_temp"
    # Per-configuration identifier used to track configs through training/logging.
    CONFIG_ID = "config_id"
    # Optional "clean" reference values, useful for noise-injection debugging
    # (the values the model *should* learn, distinct from the noisy training labels).
    ACTUAL_ENERGY = "actual_energy"
    ACTUAL_FORCES = "actual_forces"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict
