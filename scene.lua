dofile "obj_import.lua"

render {
    size = {800, 600},
    max_bounces = 3,
    antialiasing = 1
}

camera {
	eye = {0, 3, 10},
	look = {0, 0, 0}
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

ball3 = sphere {
    position = {3, 0, 0},
    radius = 1,
    color = {0.0, 0.0, 1.0},
    specular = 0.4,
    shininess = 0.01
}
