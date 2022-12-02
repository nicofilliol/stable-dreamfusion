from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import cv2
import os
import re

class AnimationButtons():
    def play_scatter(frame_duration = 500, transition_duration = 300):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": False},
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "quadratic-in-out"}}])
    
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

imgs = [nerf, noisy, denoised, residual]
titles = [f"Iteration {it}" for it in iterations]

# Create "template" figure to transfer layout onto the `fig` figure
layout = make_subplots(rows=2, cols=2, 
                       subplot_titles=["NeRF Render", "Noisy Image", "Denoised", "Residual"],
                       specs=[[{"type":"Image"}, {"type":"Image"}], [{"type":"Image"}, {"type":"Image"}]], 
                       horizontal_spacing = 0.05,
                       vertical_spacing = 0.05,
                       row_heights=[500,500], column_widths=[500,500])

max_idx = min(len(nerf), len(noisy), len(denoised), len(residual))
for i, title in enumerate(titles[:max_idx]):
    layout.add_trace(go.Image(z=nerf[i]), row=1, col=1)                   
    layout.add_trace(go.Image(z=noisy[i]), row=1, col=2)                   
    layout.add_trace(go.Image(z=denoised[i]), row=2, col=1)                   
    layout.add_trace(go.Image(z=residual[i]), row=2, col=2)                   

layout.update_layout(title="Prompt",
                     updatemenus=[dict(type="buttons", buttons=[AnimationButtons.play(), AnimationButtons.pause()])])
layout.show()