# EV1_body_yaw_B64k

Branch `refactor/evader` (uncommitted), task evader, B 65536, 1000 iterations, RCI gpufast V100.

First run of the refactored evader:
- waypoints: position uniform in a ball of radius 3 m (command.radius) around the drone + uniform random heading;
  reached within 0.2 m and 10 deg (at_target_yaw_threshold)
- no ground; termination when farther than 2 x radius from the waypoint (roll/pitch termination off)
- obs frame (23, history 4 -> 92): waypoint in body frame / radius, rotation matrix (9, body -> world, row-major),
  body lin vel (/3, clip +-3), body ang vel (/pi, clip +-3), yaw error (wrapped, /pi), last actions
- new reward yaw: 5.0 x heading progress (rad^2), like target
- actions unchanged: SRT motor commands clipped to [0, 1]
