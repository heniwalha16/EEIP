from PIL import Image, ImageDraw
import io
import base64

def drawrect(x, y):
    d_rect = True

    if (int(x) * int(y)) <= 10000:
        recWidth = int(x) * 5
        recHeight = int(y) * 5
    else:
        recWidth = int(x)
        recHeight = int(y)

    h = x 
    w = y 

    # Create a new image and draw object
    img = Image.new('RGB', (recWidth + 80, recHeight + 80), color='white')

    draw = ImageDraw.Draw(img)

    # Draw rectangle with thicker stroke
    draw.rectangle([(10, 10), (recWidth + 10, recHeight + 10)], width=2, outline='black', fill=None)
       # draw.rectangle((x1, y1, x2, y2), outline='black', width=3)

    # Calculate text position and draw the dimensions
    h_text_size = draw.textsize(h)
    w_text_size = draw.textsize(w)
    h_pos = ((recWidth + 30 - h_text_size[0]) // 2, 12)
    w_pos = (recWidth + 9 - w_text_size[0], (recHeight + 20 - w_text_size[1]) // 2)
    draw.text(h_pos, h, fill='black')
    draw.text(w_pos, w, fill='black')

    # Convert image to base64 string and return
    buffered = io.BytesIO()
    img.show()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

import math

def draw_parallelogram(width, height, angle_degrees):
    # Convert angle to radians
    angle = angle_degrees * math.pi / 180.0

    # Calculate coordinates of parallelogram vertices
    x1 = 0
    y1 = 0
    x2 = width
    y2 = 0
    x3 = width + height * math.cos(angle)
    y3 = height * math.sin(angle)
    x4 = height * math.cos(angle)
    y4 = height * math.sin(angle)

    # Create a new image and draw object
    img = Image.new('RGB', (max(int(x3)+20, int(x4)+20), max(int(y3)+20, int(y4)+20)), color='white')
    draw = ImageDraw.Draw(img)

    # Draw parallelogram
    draw.polygon([(x1+10, y1+10), (x2+10, y2+10), (x3+10, y3+10), (x4+10, y4+10)], outline='black', width=2)

    # Draw axes with labels
    draw.line([(x1+10, y1+10), (x2+10, y2+10)], fill='black', width=2)
    draw.line([(x2+10, y2+10), (x3+10, y3+10)], fill='black', width=2)
    draw.text((x2/2+10, 0), str(width), fill='black')
    draw.line([(x1+10, y1+10), (x4+10, y4+10)], fill='black', width=2)
    draw.line([(x4+10, y4+10), (x3+10, y3+10)], fill='black', width=2)
    draw.text((x4/2-5, y4/2+10), str(height), fill='black')

    # Convert image to base64 string and return
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


    import io
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
def draw_circle(radius):
    # Clear the current figure
    plt.clf()
    
    # Create a new figure with larger size
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw circle with thicker stroke
    circle = Circle((radius + 40, radius + 40), radius, edgecolor='black', facecolor='none', linewidth=4)
    ax.add_patch(circle)
    
    # Draw the radius line
    center = (radius + 40, radius + 40)
    end = (center[0] + radius, center[1])
    ax.plot([center[0], end[0]], [center[1], end[1]], color='black', linewidth=4)

    # Set axis limits and turn off the axis ticks
    ax.set_xlim([0, 2*radius+80])
    ax.set_ylim([0, 2*radius+80])


    # Draw the radius value
    text = f"Radius = {radius}"
    ax.text((2*radius+80)/2, 10, text, ha='center')

    # Save figure to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    # Convert buffer to base64 string and return
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


import matplotlib.pyplot as plt

def draw_rhombus(diagonal1, diagonal2):
    # Calculate the half-lengths of each diagonal
    half_d1 = diagonal1 / 2
    half_d2 = diagonal2 / 2

    # Create a new plot and set the axis limits
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(diagonal1, diagonal2) + 20)
    ax.set_ylim(0, max(diagonal1, diagonal2) + 20)

    # Calculate the coordinates of the four vertices
    x1 = 0
    y1 = half_d2
    x2 = half_d1
    y2 = 0
    x3 = diagonal1
    y3 = half_d2
    x4 = half_d1
    y4 = diagonal2

    # Draw the rhombus and the line from the center to one midpoint
    ax.plot([x1+10, x2+10, x3+10, x4+10, x1+10], [y1+10, y2+10, y3+10, y4+10, y1+10], color='black', linewidth=2)
    ax.plot([half_d1+10, x2+10], [half_d2+10, y2+10], color='black', linewidth=2)

    # Add labels for the diagonals
    ax.text(half_d1/2+10, half_d2+5, str(diagonal2), ha='center', va='bottom')
    ax.text(half_d1+5, diagonal2/2+10, str(diagonal1), ha='left', va='center')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')

    # Convert buffer to base64 string and return
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def metric_conversion(l):
    available_units = ('Millimetre', 'Centimetre', 'Decimetre','Metre', 'Kilometre')
    conversions = (1, 10,100, 1000, 1e6)
    index = 0
    unit = l[1]
    num2=l[0]
    for i in range (0, len(available_units)):
        if available_units[i] == str(l[1]):
            num_in_mm = l[0] * conversions[i]
            break
    for j in range (0, len(available_units)):
        if ((num_in_mm / conversions[j])>9.99) and ((num_in_mm / conversions[j])<1000.01) :
            unit=available_units[j]
            num2 = num_in_mm / conversions[j]
            break
    return [num2,l[0],l[1]]

                