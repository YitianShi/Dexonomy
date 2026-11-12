"""Small MjSpec compatibility helpers for current and archived environments."""


def delete_spec_element(spec, element):
    """Use the owning spec's deletion API, introduced after MuJoCo 3.3.3."""
    if hasattr(spec, "delete"):
        spec.delete(element)
    else:
        element.delete()
