from PIL import Image
from demo import visual_grounding
img = Image.open("demo/vases.jpg")
overlay, mask = visual_grounding(img, "the vase with the two white roses")
Image.fromarray(overlay).save("out.png")