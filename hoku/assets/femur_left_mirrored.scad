% scale(1000) import("femur_left_mirrored.stl");

// Sketch femur_left_mirrored_cylinder 300
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 300.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=55.000000,h=thickness);
  }
}
}
