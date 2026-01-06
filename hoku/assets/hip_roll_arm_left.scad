% scale(1000) import("hip_roll_arm_left.stl");

// Sketch hip_roll_arm_left_cylinder 67.5
multmatrix([[0.0, 0.0, 1.0, 9.999999999999998], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 67.500000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=45.000000,h=thickness);
  }
}
}
