"""Human teleoperation of the direction task's commands, via the Viser viewer."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import viser

__all__ = ["TeleopState", "build_teleop_gui", "teleop_state"]

_ROTATE_STEP_DEG = 15.0

_HotkeyKey = Literal["I", ",", "J", "L", "U", "O", "K"]

_TURN_OPTIONS = ("Left", "None", "Right")
_TURN_VALUES = {"Left": 1.0, "None": 0.0, "Right": -1.0}


@dataclass
class TeleopState:
    enabled: bool = False
    """Master gate."""
    move: bool = False
    """False means heading ``<0, 0>`` -- a zero command."""
    heading_deg: float = 0.0
    """Commanded heading in the robot's base frame, in [-180, 180]. 0 is straight ahead."""
    turn: float = 0.0
    """Discrete turn command in {-1, 0, +1}."""

    def zero(self) -> None:
        """Return to a plain standing."""
        self.move = False
        self.heading_deg = 0.0
        self.turn = 0.0


teleop_state = TeleopState()
"""Process-wide teleop state."""


def _wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def heading_vector(state: TeleopState) -> tuple[float, float]:
    if not state.move:
        return 0.0, 0.0
    psi = math.radians(state.heading_deg)
    return math.cos(psi), math.sin(psi)


def build_teleop_gui(
    server: viser.ViserServer,
    state: TeleopState,
    on_change: Callable[[], None] | None = None,
) -> None:
    """Build the teleop panel and register its hotkeys.

    Data flows one way: hotkeys assign to widget handles, widget callbacks are the only
    thing that mutates *state*. Assigning to a handle's ``value`` invokes its callbacks
    synchronously, so the panel and the state can never disagree.
    """
    from viser import Icon

    def changed() -> None:
        if on_change is not None:
            on_change()

    with server.gui.add_folder("Teleop"):
        enable = server.gui.add_checkbox("Enable", initial_value=False)
        move = server.gui.add_checkbox("Move", initial_value=False)
        heading = server.gui.add_slider("Heading (deg)", min=-180.0, max=180.0, step=5.0, initial_value=0.0)
        turn = server.gui.add_button_group("Turn", options=list(_TURN_OPTIONS))
        stand_btn = server.gui.add_button("Stand", icon=Icon.SQUARE_X)

    @enable.on_update
    def _(_) -> None:
        state.enabled = enable.value
        changed()

    @move.on_update
    def _(_) -> None:
        state.move = move.value
        changed()

    @heading.on_update
    def _(_) -> None:
        state.heading_deg = float(heading.value)
        changed()

    @turn.on_click
    def _(_) -> None:
        state.turn = _TURN_VALUES.get(turn.value, 0.0)
        changed()

    def stand() -> None:
        move.value = False
        heading.value = 0.0
        turn.value = "None"

    @stand_btn.on_click
    def _(_) -> None:
        stand()

    # -- Hotkeys

    def _command(label: str, key: _HotkeyKey, fn: Callable[[], None]) -> None:
        handle = server.gui.add_command(label, hotkey=cast("Any", key))

        @handle.on_trigger
        def _(_) -> None:
            fn()

    def _face(deg: float) -> None:
        heading.value = deg
        move.value = True

    def _rotate(delta: float) -> None:
        heading.value = _wrap_deg(float(heading.value) + delta)
        move.value = True

    def _step_turn(delta: float) -> None:
        current = _TURN_VALUES.get(turn.value, 0.0)
        turn.value = _TURN_OPTIONS[1 - int(max(-1.0, min(1.0, current + delta)))]

    _command("Teleop: forward", "I", lambda: _face(0.0))
    _command("Teleop: reverse", ",", lambda: _face(180.0))
    _command("Teleop: steer left", "J", lambda: _rotate(_ROTATE_STEP_DEG))
    _command("Teleop: steer right", "L", lambda: _rotate(-_ROTATE_STEP_DEG))
    _command("Teleop: turn left", "U", lambda: _step_turn(1.0))
    _command("Teleop: turn right", "O", lambda: _step_turn(-1.0))
    _command("Teleop: stand", "K", stand)
