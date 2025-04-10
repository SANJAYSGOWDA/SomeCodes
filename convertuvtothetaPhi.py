def pixelToSpherical(x, y, w, h):
    print(x,y)
    fovH = 360 
    fovV = 180 
    xNormalized = (x - w / 2) / (w / 2)
    yNormalized = (y - h / 2) / (h / 2)
    
    theta = xNormalized * (fovH / 2)  # Azimuth (horizontal angle)
    phi = yNormalized * (fovV / 2)  # Elevation (vertical angle)
    
    return theta, phi

if __name__ == '__main__':
    imageWidth = 5000
    imageHeight = 2700
    u = 2043
    v = 1378

    thetas, phis = pixelToSpherical(u, v, imageWidth, imageHeight)
    print("thetas", thetas)
    print("phis", phis)
