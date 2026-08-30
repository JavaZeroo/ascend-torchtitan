"""Training entry point: ``python -m ascend_titan.train --module ... --config ...``.

Identical to ``python -m torchtitan.train`` except that :func:`ascend_titan.setup`
runs first. That ordering is the whole point of this file (see _bootstrap.py).
"""


def main() -> None:
    import ascend_titan

    # A training entry without a device backend is a broken environment, not a degraded
    # one (ADR-004 covers kernel dependencies, not the backend): fail here, loudly.
    ascend_titan.setup()

    # Imported only now, after setup(): torchtitan freezes device_type on import.
    from torchtitan.train import main as titan_main

    titan_main()


if __name__ == "__main__":
    main()
