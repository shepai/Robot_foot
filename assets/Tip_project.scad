outer_r = 40;
wall = 2;
inner_r = outer_r - wall;

num_bumps = 133;
golden_angle = 137.50776405003785;

difference() {

    // -----------------------------
    // OUTER SHELL (hemisphere)
    // -----------------------------
    intersection() {
        sphere(r = outer_r);
        translate([-outer_r, -outer_r, 0])
            cube([outer_r * 2, outer_r * 2, outer_r]);
    }

    difference() {

        // -----------------------------
        // INNER CAVITY (hemisphere)
        // -----------------------------
        intersection() {
            sphere(r = inner_r);
            translate([-outer_r, -outer_r, 0])
                cube([outer_r * 2, outer_r * 2, outer_r]);
        }

        // -----------------------------
        // UNIFORMLY DISTRIBUTED BUMPS
        // (Fibonacci sphere)
        // -----------------------------
        for (i = [0:num_bumps - 1]) {

            // z from 1 to -1 (uniform)
            z = 1 - 2 * (i + 0.5) / num_bumps;

            radius = sqrt(1 - z * z);
            theta = i * golden_angle;

            x = (inner_r - 1) * radius * cos(theta);
            y = (inner_r - 1) * radius * sin(theta);
            zz = (inner_r - 1) * z;

            translate([x, y, zz])
                sphere(r = 1.2);
        }
    }
}