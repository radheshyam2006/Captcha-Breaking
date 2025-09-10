from PIL import Image, ImageDraw, ImageFont
import random
import os
import csv

word = [
    "abstract", "account", "action", "advice", "agenda", "algorithm", "analysis", "answer",
    "argument", "arrival", "aspect", "association", "attitude", "balance", "belief", "benefit",
    "boundary", "career", "category", "challenge", "choice", "concept", "context", "culture",
    "debate", "decision", "definition", "detail", "development", "discussion", "education",
    "effort", "emotion", "energy", "engine", "environment", "evidence", "example", "experience",
    "factor", "feature", "focus", "framework", "abcd", "efgh", "ijkl", "mnop", "qrst", "uvwxyz",
    "planet", "guitar", "engine", "tunnel", "horizon", "puzzle", "lantern", "comet", "bridge", "marble",
    "voyage", "whistle", "canvas", "chimney", "glacier", "fountain", "jungle", "satellite", "compass", "blueprint",
    "cactus", "ribbon", "volcano", "parachute", "echo", "labyrinth", "radar", "pedal", "quartz", "beacon",
    "harvest", "scroll", "zeppelin", "spectrum", "pyramid", "nebula", "crescent", "catapult", "plasma", "zipper",
    "meadow", "telescope", "pendulum", "monsoon", "alchemy", "gargoyle", "snowflake", "hourglass", "firefly", "reef"
]

DIR=input("Enter the directory name: ")
csv_file = os.path.join(DIR, "labels.csv")
os.makedirs(DIR, exist_ok=True)
easynumber = int(input("Enter the number of easy images to generate: "))
hdata = int(input("Enter the number of hard images to generate: "))

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

# Add noise to the image
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

c=0
fonts="arial.ttf"
font = ImageFont.truetype(fonts, size=32)

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ImageName", "TextLabel"])# Header row
    for i in range(easynumber):
        img = Image.new("RGB", (250, 75), color="white")
        painter = ImageDraw.Draw(img)
        w = random.choice(word)  # Select a random word
        w = w.capitalize()
        # Draw text on image

        painter.text((5, 5), w, font=font, fill="black")

        # Save image
        img_name = f"{c}.png"
        c=c+1
        img.save(os.path.join(DIR, img_name))

        # Write to CSV and flush
        writer.writerow([img_name, w])
        file.flush()  # Ensure writing happens immediately

       






fonts = [
    ImageFont.truetype("arial.ttf", 32),
    ImageFont.truetype("times.ttf", 32),
    ImageFont.truetype("comic.ttf", 32)
]
fillcolor = ["black", "red", "blue", "green", "orange", "purple", "brown", "grey"]
n = [0, 5, 10, 15, 20, 30]

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(["ImageName", "TextLabel"])  # Header row

    for i in range(hdata):
        w = random.choice(word)
        font = random.choice(fonts)
        noise = random.choice(n)
        bg = "#f0f0f0"
        img = Image.new("RGB", (250, 75), color=bg)
        painter = ImageDraw.Draw(img)
        fword = capital(w)
        col = random.choice(fillcolor)
        painter.text((5, 5), fword, font=font, fill=col)
        fimg = add_noise(img, noise)
        img_name = f"{c}.png"
        c=c+1
        fimg.save(os.path.join(DIR, img_name))
        writer.writerow([img_name, fword])


print("Image generation and CSV writing complete.")
