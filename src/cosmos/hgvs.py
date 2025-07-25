"""
Parses missense/synonymous mutations in HGVS nomenclature.
"""

from dataclasses import dataclass


@dataclass
class Missense:
    """
    Parses a missense mutation in the format `p.(I328P)`.
    """

    wildtype: str
    position: int
    mutant: str

    @classmethod
    def from_hgvs(cls, hgvs: str):
        """
        e.g. `p.(I328P)`
        """
        assert hgvs.startswith("p.(") and hgvs.endswith(")")
        hgvs = hgvs[3:-1]
        return cls(
            wildtype=hgvs[0],
            position=int(hgvs[1:-1]),
            mutant=hgvs[-1],
        )

    def is_synonymous(self) -> bool:
        """
        Check if the mutation is synonymous.
        """
        return self.wildtype == self.mutant
