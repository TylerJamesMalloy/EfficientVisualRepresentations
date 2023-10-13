import cairo

paths = ["raw/orange.png", "raw/green.png", "raw/purple.png"]
for idx, colors in enumerate([[0.75, 0.5, 0.0], [0.5, 0.75, 0], [0.5, 0, 0.75]]):
    path = paths[idx]
    # Create a PNG surface and a context
    width, height = 64, 64
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
    context = cairo.Context(surface)

    context.set_source_rgb(1,1,1) # transparent white
    context.rectangle(0, 0, 64, 64)
    context.fill()

    # Set the fill color to the color
    #context.set_source_rgb(colors[0], colors[1], colors[2])  # RGB values for the color

    # Draw a circle in the center of the surface
    center_x, center_y = width / 2, height / 2
    radius = min(center_x, center_y) - 1  # 10-pixel padding
    border_width = 3

    # Draw the circle border (black)
    context.set_source_rgb(0.0, 0.0, 0.0)  # RGB values for black
    context.set_line_width(border_width)
    context.arc(center_x, center_y, radius, 0, 2 * 3.141592)  # 2 * π
    context.stroke()

    # Fill the circle with the color
    context.set_source_rgb(colors[0], colors[1], colors[2])  # RGB values for the color
    context.arc(center_x, center_y, radius - border_width + 2, 0, 2 * 3.141592)  # 2 * π
    context.fill()

    # Save the result to a PNG file
    surface.write_to_png(path)


