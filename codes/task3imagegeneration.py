from PIL import Image, ImageDraw, ImageFont
import random
import os
import csv

word="abcdefghijklmnopqrstuvwxyz"
maxi=10

DIR=input("Enter the directory name: ")
csv_file=os.path.join(DIR, "labels.csv")
os.makedirs(DIR, exist_ok=True)

def capital(text):
    if not text:  
        return text

    indices = [i for i, char in enumerate(text) if char.isalpha()]
    if not indices:
        return text

    num_to_capitalize = random.randint(1, len(indices))
    chosen_indices = random.sample(indices, num_to_capitalize)
    text = ''.join(
    (char.upper() if i in chosen_indices else char) 
    for i, char in enumerate(text)
    )

    return text

def add_noise(image, intensity):
    pixels = image.load()
    for i in range(image.height):
        for j in range(image.width):
            r, g, b = pixels[j, i]
            noise = random.randint(-intensity, intensity)
            pixels[j, i] = (max(0, min(255, r + noise)),
                            max(0, min(255, g + noise)),
                            max(0, min(255, b + noise)))
    return image


fonts="arial.ttf"
font = ImageFont.truetype(fonts, size=32)
bg=["green","red"]

c=0
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    easynumber=int(input("Enter the number of images to generate: "))
    for i in range(easynumber):
        w=""
        col=random.choice(bg)
        img = Image.new("RGB", (450, 75), color=col)
        painter = ImageDraw.Draw(img)
        leng = random.randint(1, maxi)
        for _ in range(leng):
            w+=random.choice(word)
        
        w = capital(w)
        
        
        painter.text((5, 5), w, font=font, fill="black")
        

        img_name = f"{c}.png"
        c=c+1
        img.save(os.path.join(DIR, img_name))
        r=''

        writer.writerow([img_name, w])
        file.flush()  


print("Image generation and CSV writing complete.")



