from PIL import Image

img = Image.open('/image/resize.jpeg')
longer_side = max(img.size)
horizontal_padding = (longer_side - img.size[0]) / 2
vertical_padding = (longer_side - img.size[1]) / 2
img = img.crop(
    (
        -horizontal_padding,
        -vertical_padding,
        img.size[0] + horizontal_padding,
        img.size[1] + vertical_padding
    )
)
img = img.resize((32, 32), Image.ANTIALIAS)
img.save('/tmp/resize_out.jpg')