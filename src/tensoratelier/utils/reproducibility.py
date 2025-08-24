from dataclasses import dataclass


@dataclass
class _Seed:
    _seed: int = None

    @property
    def get_seed(self) -> bool:
        if self._seed is None:
            raise NameError("No seed set")
        return self._seed

    def set_seed(self, value) -> bool:
        if not isinstance(value, int):
            raise TypeError("Seed must be an integer")

        self._seed = value

    @property
    def is_set(self) -> bool:
        if self._seed is not None:
            return True
        return False


seed = _Seed()
