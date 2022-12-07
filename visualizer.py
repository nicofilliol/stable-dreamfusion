from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import cv2
import os
import re
import xarray as xr
import pandas as pd

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

    all_iterations = list(map(lambda f: int(re.sub('\D', '', f)), files))
    iterations = []
    for i in range(0, len(files), 5):
        img = cv2.imread(f'{folder}/{files[i]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        iterations.append(all_iterations[i])

    return imgs, iterations 


nerf, iterations = load_images("visualizations/front/nerf")
noisy, _ = load_images("visualizations/front/noisy")
denoised, _ = load_images("visualizations/front/final_denoised")
residual, _ = load_images("visualizations/front/residual")

n = min(len(nerf), len(noisy), len(denoised), len(residual))

imgs = np.array(list(zip(nerf, noisy, denoised, residual))) # order is flipped
titles = [f"<b>Iteration {it}</b>" for it in iterations]
columns = ["Denoised", "Residual", "NeRF", "Noisy",] # weird convention

print(imgs.shape)
data = xr.DataArray(np.array(imgs), coords=dict(iterations=iterations, img_type=np.arange(4)), dims=["iterations", "img_type", "height", "width", "rgb"])
print(data)
#print(data[0])
# Create figure on the basis of the animated facetted imshow figure
fig = px.imshow(data, facet_col="img_type", facet_col_wrap=2, width=600, height=600, animation_frame="iterations", binary_string=True, labels=dict(animation_frame="Iteration"))
for i, a in enumerate(fig.layout.annotations):
    print(a.text)
    a.text = f"<b>{columns[i]}</b>"

fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False, visible=False)
fig.update_yaxes(showticklabels=False, visible=False)
fig.show()

fig.write_html('outputs/plotly_demo_1.html')