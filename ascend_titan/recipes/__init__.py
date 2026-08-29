"""L3: Ascend recipes = upstream config + deltas.

Every recipe calls the upstream ``config_registry`` function and mutates the
returned ``Trainer.Config``; it never rebuilds the config from scratch, so new
upstream fields are inherited automatically.

Run with:  python -m ascend_titan.train --module ascend_titan.recipes.<model> --config <fn>
"""
