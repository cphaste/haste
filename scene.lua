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

floor = plane {
    color = {1.0, 1.0, 1.0},
    distance = -1
}

ball = sphere {
    position = {-3, 0, 0},
    radius = 1,
    color = {1.0, 0.0, 0.0},
    specular = 0.4,
    shininess = 0.01
}

tri = triangle {
    color = {0.0, 1.0, 0.0},
    normal1 = {1, -1, 1},
    normal2 = {0, 1, 1},
    normal3 = {-1, -1, 1}
}

ball3 = sphere {
    position = {3, 0, 0},
    radius = 1,
    color = {0.0, 0.0, 1.0},
    specular = 0.4,
    shininess = 0.01
}
