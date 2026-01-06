% scale(1000) import("torso_yaw_fork.stl");

// Sketch torso_yaw_fork_cylinder 67.5
multmatrix([[0.0, 0.0, 1.0, 10.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 67.500000;
translate([0, 0, -thickness]) {
  translate([-27.250000, 0.000000, 0]) {
    cylinder(r=35.000000,h=thickness);
  }
}
}
