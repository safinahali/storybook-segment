import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys


#example image
image = cv2.imread('images/story.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#display functions for markers
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.7])], axis=0)
    else:
        color = np.array([30/255, 255/255, 144/255, 0.7])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    cv2.imwrite('mask.jpg', mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


#select objects with SAM 

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

input_point = np.array([[120, 260]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

masks.shape  # (number_of_masks) x H x W

# variables
ix = -1
iy = -1
drawing = False
  
def draw_rectangle_with_drag(event, x, y, flags, param):
      
    global ix, iy, drawing, image
      
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y            
              
      
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(image, pt1 =(ix, iy),
                      pt2 =(x, y),
                      color =(0, 255, 255),
                      thickness =2)
        alpha = 0.4 

        print(ix, iy, x, y)
        drawing = False
        
#         This line for deciding the input box
        input_box = np.array([ix, iy, x, y])

        masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,)

        #we want to save masks

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        show_mask(masks[0], plt.gca())
        # show_box(input_box, plt.gca())
        plt.axis('off')
        plt.show()
          
cv2.namedWindow(winname = "Draw a rectangle over the desired area")
cv2.setMouseCallback("Draw a rectangle over the desired area", 
                     draw_rectangle_with_drag)
  
while True:
    cv2.imshow("Draw a rectangle over the desired area", image)
      
    if cv2.waitKey(0) & 0xFF == ord('x'):
        print("quitting now")
        break
        cv2.destroyAllWindows()
        quit()
        exit_program()

def exit_program():
    print("Exiting the program...")
    sys.exit(0)


    



