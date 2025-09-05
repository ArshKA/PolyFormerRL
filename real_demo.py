from PIL import Image
from demo import visual_grounding
img = Image.open("demo/dog.jpg")
overlay, mask = visual_grounding(img, "the german shepard")
Image.fromarray(overlay).save("out.png")