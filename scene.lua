render {
    size = {1024, 768},
    max_bounces = 3,
    antialiasing = 3
}

camera {
	eye = {0, 3, 10}
}

lt = light {
    -- using defaults for now
}

ball = sphere {
    position = {-3, 0, 0},
    radius = 1,
    color = {1.0, 0.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ball2 = sphere {
    position = {0, 0, 3},
    radius = 1,
    color = {0.0, 1.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ball3 = sphere {
    position = {3, 0, 0},
    radius = 1,
    color = {0.0, 0.0, 1.0},
    specular = 0.4,
    shininess = 0.01
}
