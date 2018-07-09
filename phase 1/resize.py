from PIL import Image
import glob
basewidth = 256
test_images = glob.glob('images/*.jpg')

    # Step through the test images
for filename in test_images:
    img = Image.open(filename)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(filename) 
