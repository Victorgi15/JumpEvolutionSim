Minimal 2D triad simulation.

The project focuses on one controlled triad: three particles, two rigid links,
and a spring-damper angle controller at the center joint.

Run the visual triad demo:

    python -m sim.visualize_triad

Run the visual tetrad demo:

    python -m sim.visualize_tetrad

Try the first random tetrad population:

    python -m sim.evolve_tetrad

Adjust the tetrad first position pivot target angles:

    python -m sim.visualize_tetrad --angle1 72 --angle2 138

The tetrad view exposes three positions. Each position stores one target angle
per pivot, and the start button loops through the three positions. The clock
knob controls the playback speed; the default is 0.5 Hz, so each position lasts
two seconds. The HUD also shows the actuator energy accumulated since the start
of the simulation, the current actuator power, and a movement score. The score
rewards horizontal center-of-mass displacement and subtracts an energy penalty.

Run the tests:

    pytest

Pygame is optional. Without it, the viewer prints a short console trace.
