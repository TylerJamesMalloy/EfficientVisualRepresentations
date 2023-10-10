import cairo

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 64, 64)
ctx = cairo.Context(surface)

ctx.rectangle(0, 0, 64, 64)
ctx.set_source_rgb(0.75, 0.5, 0) # orange
ctx.fill()
surface.write_to_png('raw/orange.png')

ctx = cairo.Context(surface)

ctx.rectangle(0, 0, 64, 64)
ctx.set_source_rgb(0.5, 0.75, 0) # purple
ctx.fill()
surface.write_to_png('raw/green.png')

ctx = cairo.Context(surface)

ctx.rectangle(0, 0, 64, 64)
ctx.set_source_rgb(0.5, 0, 0.75) # green
ctx.fill()
surface.write_to_png('raw/purple.png')
