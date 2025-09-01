from typing import Dict, Set, Any

class AttributeOverrideMixin:

    _attribute_overrides: Dict[str, str] = {
        'optimizer': 'optimizer',
    }

    def __setattr__(self, name: str, value: Any) -> None:

        if name in self._attribute_overrides:
            if name in self._attribute_overrides:

                private_name = self._attribute_overrides[name]
                object.__setattr__(self, private_name, value)

            else:
                super().__setattr__(name, value)

        
    def __getattribute__(self, name: str) -> Any:

        try:
            overrides = object.__getattribute__(self, '_attribute_overrides')
            if name in overrides:
                private_name = overrides[name]
                return object.__getattribute__(self, private_name)

        except AttributeError:
            pass

        return super().__getattribute__(name)
