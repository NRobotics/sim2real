% scale(1000) import("hip_roll_arm_left_mirrored.stl");

// Sketch hip_roll_arm_left_mirrored_cylinder 67.5
multmatrix([[0.0, 0.0, -1.0, -57.499999999999986], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 67.500000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=45.000000,h=thickness);
  }
}
}
