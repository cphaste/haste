render {
    size = {1024, 768},
    max_bounces = 3,
    antialiasing = 3
}

camera {
	eye = {0, 4, 10},
	look = {0, 1, 0}
}

lt = light {
    -- using defaults for now
}

ball = sphere {
    position = {-3, 1, 0},
    radius = 1,
    color = {1.0, 0.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ball2 = sphere {
    position = {0, 1, 3},
    radius = 1,
    color = {0.0, 1.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

ball3 = sphere {
    position = {3, 1, 0},
    radius = 1,
    color = {0.0, 0.0, 1.0},
    specular = 0.4,
    shininess = 0.01
}

floor = plane {
    color = {0.85, 0.85, 0.85}
}
