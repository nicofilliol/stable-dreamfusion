from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import cv2
import os
import re
import xarray as xr

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
columns = ["NeRF", "Noisy", "Denoised", "Residual"]

data = xr.DataArray(np.array(imgs), dims=("x", "nerf", "noisy", "denoised", "residual"), coords={"x": iterations})
# Create figure on the basis of the animated facetted imshow figure
fig = px.imshow(data, facet_col=1, facet_col_wrap=2, width=600, height=600, animation_frame=0, binary_string=True, labels=dict(animation_frame="Iteration"))
for i, a in enumerate(fig.layout.annotations):
    a.text = f"<b>{columns[i]}</b>"

fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()

fig.write_html('outputs/plotly_demo_1.html')