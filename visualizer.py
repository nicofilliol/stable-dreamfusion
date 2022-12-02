from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import cv2
import os
import re

class AnimationButtons():
    def play(frame_duration = 1000, transition_duration = 0):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "mode":"immediate",
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    def pause():
        return dict(label="Pause", method="animate", args=
                    [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])


def load_images(folder) -> list:
    imgs = []
    # Load in the images
    files = os.listdir(folder)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    iterations = list(map(lambda f: int(re.sub('\D', '', f)), files))

    for filepath in files:
        img = cv2.imread(f'{folder}/{filepath}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    return imgs, iterations 


nerf, iterations = load_images("outputs/visualizations/front/nerf")
noisy, _ = load_images("outputs/visualizations/front/noisy")
denoised, _ = load_images("outputs/visualizations/front/denoised")
residual, _ = load_images("outputs/visualizations/front/residual")

imgs = list(zip(nerf, noisy, denoised, residual))
titles = [f"<b>Iteration {it}</b>" for it in iterations]

# Create figure on the basis of the animated facetted imshow figure
fig = px.imshow(np.array(imgs), facet_col=1, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))

for i, title in enumerate(titles):
    fig["frames"][i]["layout"]["title"] = title 

# Create "template" figure to transfer layout onto the `fig` figure
layout = make_subplots(rows=2, cols=2, 
                       subplot_titles=["NeRF Render", "Noisy Image", "Denoised", "Residual"],
                       specs=[[{"type":"Image"}, {"type":"Image"}], [{"type":"Image"}, {"type":"Image"}]],
                       row_heights=[250, 250],
                       column_widths=[250, 250],
                       vertical_spacing=0.075,
                       horizontal_spacing=0.05)

layout.update_layout(title="<b>Iteration 0</b>",
                     title_x=0.5,
                     width= 600,
                     updatemenus=[dict(type="buttons", buttons=[AnimationButtons.play(), AnimationButtons.pause()], direction="left", x = 0.5, xanchor = 'center', y = -0.1, yanchor = 'bottom')],
                     xaxis_visible=False, yaxis_visible=False,
                     xaxis2_visible=False, yaxis2_visible=False,
                     xaxis3_visible=False, yaxis3_visible=False,
                     xaxis4_visible=False, yaxis4_visible=False)



fig["layout"] = layout["layout"]
fig.show()