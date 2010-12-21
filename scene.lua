-- import obj mesh type
require "helpers/obj"

-- render configuration
render {
    size = {800, 600},
    max_bounces = 2,
    antialiasing = 3,
    direct_samples = 32,
    gamma_correction = 1 / 2.2
}

-- camera setup
camera {
	eye = {0.75, 0.5, 1},
	look = {0, 0.2, 0}
}

-- sphere area light
arealt = sphere {
    position = {-1, 5, 3},
    radius = 2.5,
    color = {1, 1, 1},
    emissive = 1,
    ambient = 0,
    diffuse = 0,
    specular = 0
}

-- scene floor
triangle {
    color = {0.3, 0.3, 0.3},
    vertex1 = {-10, -0.5, 10},
    vertex2 = {-10, -0.5, -10},
    vertex3 = {10, -0.5, -10},
    normal1 = {0, 1, 0},
    normal2 = {0, 1, 0},
    normal3 = {0, 1, 0}
}
triangle {
    color = {0.3, 0.3, 0.3},
    vertex1 = {10, -0.5, -10},
    vertex2 = {10, -0.5, 10},
    vertex3 = {-10, -0.5, 10},
    normal1 = {0, 1, 0},
    normal2 = {0, 1, 0},
    normal3 = {0, 1, 0}
}

-- mesh
head = obj {
    mesh = "head.obj",
    color = {0.93725, 0.81569, 0.81176},
    specular = 0.4,
    shininess = 0.01
}
