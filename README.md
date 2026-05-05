Minimal 2D triad simulation.

The project focuses on one controlled triad: three particles, two rigid links,
and a spring-damper angle controller at the center joint.

Run the visual triad demo:

    python -m sim.visualize_triad

Run the visual tetrad demo:

    python -m sim.visualize_tetrad

Try the first random tetrad population:

    python -m sim.evolve_tetrad

Run several evolution generations:

    python -m sim.evolve_tetrad --population 1000 --generations 20 --duration 10 --seed 42

Each generation logs its top three candidates to logs/evolve_tetrad.jsonl by
default. New random immigrants are injected at each generation to preserve
exploration; the default immigrant ratio is 10%. Position-count mutations can
add or remove movement poses between the default bounds of 2 and 8 positions.
Random immigrants sample the full configured space, including position count,
pivot count, angles, clock, and branch attachments.

Adjust the tetrad first position pivot target angles:

    python -m sim.visualize_tetrad --angle1 72 --angle2 138

The tetrad view exposes movement positions. Each position stores one target angle
per pivot, and the start button loops through the positions. The cycle knob
controls full-cycle playback speed; the default is 0.5 Hz, so one full movement
cycle lasts two seconds no matter how many positions exist. The HUD also shows
the actuator energy accumulated since the start of the simulation, the current
actuator power, and a movement score. The score
rewards horizontal center-of-mass displacement and subtracts energy and airborne
time penalties.

Run the tests:

    pytest

Pygame is optional. Without it, the viewer prints a short console trace.
