# Pick Ping-Pong Ball Minimal Scene

Minimal sim-to-real demo layout for the custom-arm transition.

## Physical Setup

- Table height: about 0.70 m in the real setup.
- Table surface: white by default.
- Robot mount: SO101 is placed on the front table edge, centered between the ball and target square.
- Camera background: darker neutral matte rear, left, and right panels around the table.
- Camera: INNO-MAKER U20CAM-1080P-S1, used at 640x480.
- Lighting: moderately bright indoor lighting, intentionally softer than the white-void default.

## Task Layout

The sim uses the robot-base frame as the working coordinate frame. The table top is modeled at
`z = 0.0` so the robot can sit at the tabletop plane, while the real table height is documented
above for hardware setup. Primitive assets are generated directly from the task config under each
IsaacLab env root, not from a static USD scene file.
The table visual spans the full surface, while its collision box is trimmed away from the front
edge so the temporary SO101 base does not get pushed upward by the tabletop collider.

- Green ping-pong ball:
  - radius: 0.02 m
  - mass: 0.0027 kg
  - default local position: `(0.22, -0.04, 0.02)` so it appears on the left side of the table view.
  - reset randomization: +/- 0.01 m in x/y
- Black tape square:
  - size: 0.15 m x 0.15 m
  - default local center: `(0.48, -0.04)` so it appears on the right side of the table view.
  - visual marker only

Task instruction:
`Pick up the green ping pong ball and place it inside the black square, then reset the arm to rest state.`
