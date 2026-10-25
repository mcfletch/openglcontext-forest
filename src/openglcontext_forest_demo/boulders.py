"""Boulders thrown into the forest.

A boulder is a stone wearing the forest's rock material, in the scene and a
dynamic body in the context's physics world. It meets the ground, the trunks and
the rocks the walker meets, because they are bodies in the same world. At most
``limit`` boulders are kept: throwing another removes the oldest.
"""
from __future__ import annotations

import collections
from typing import Any

import numpy as np
from OpenGLContext import entropy
from OpenGLContext.scenegraph import natural, surfaces

__all__ = ['Boulders']

#: How many boulders are kept.
LIMIT = 12

#: The radius of a boulder before it is shaped, in metres.
RADII = (0.22, 0.42)

#: How fast a boulder leaves the hand, in metres a second.
SPEED = 14.0


class Boulders:
    """The boulders thrown into ``scene``, a node that holds them, in ``physics``'s world.

    ``material`` is what each boulder wears. ``limit`` is how many are kept.
    """

    def __init__(self, physics: Any, scene: Any, material: Any, limit: int = LIMIT) -> None:
        self.physics = physics
        self.scene = scene
        self.material = material
        self.limit = int(limit)
        self._thrown: collections.deque[Any] = collections.deque()

    def __len__(self) -> int:
        return len(self._thrown)

    def throw(self, origin: Any, direction: Any, speed: float = SPEED) -> Any:
        """Throw a new boulder from ``origin`` along ``direction`` at ``speed`` metres a second.

        Returns its body. Raises ValueError for a ``direction`` of no length.
        """
        heading = np.asarray(direction, dtype='d')
        length = float(np.linalg.norm(heading))
        if length < 1e-9:
            raise ValueError(f'a boulder is thrown along a direction, not {heading.tolist()!r}')
        rng = entropy.generator('boulders')
        radius = float(rng.uniform(*RADII))
        stone = natural.stone(rng, (radius, radius * rng.uniform(0.75, 1.0),
                                    radius * rng.uniform(0.9, 1.2)), divisions=2)
        node = surfaces.shape(stone, self.material,
                              translation=tuple(float(v) for v in origin))
        self.scene.children.append(node)
        body = self.physics.add_dynamic(node, proxy='sphere', restitution=0.25, friction=0.8)
        body.set_velocity(heading / length * float(speed))
        self._thrown.append(body)
        while len(self._thrown) > self.limit:
            self._drop(self._thrown.popleft())
        return body

    def _drop(self, body: Any) -> None:
        """Take a boulder out of the world and the scene."""
        body.remove()
        if body.node in self.scene.children:
            self.scene.children.remove(body.node)
