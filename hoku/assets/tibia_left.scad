% scale(1000) import("tibia_left.stl");

// Sketch tibia_left_cylinder 350
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -7.401486830834379e-17, -1.2582527612418445e-14], [0.0, 7.401486830834379e-17, 1.0, 170.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 350.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=40.000000,h=thickness);
  }
}
}
