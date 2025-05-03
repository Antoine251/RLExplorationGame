import imageio.v3 as iio
import numpy as np

frames = np.vstack(
    [
        iio.imread(
            "/Users/antoinelaborde/Downloads/Demo_RL_game-ezgif.com-video-to-gif-converter.gif"
        ),
        iio.imread(
            "/Users/antoinelaborde/Downloads/Demo_RL_game-ezgif.com-video-to-gif-converter (1).gif"
        ),
        iio.imread(
            "/Users/antoinelaborde/Downloads/Demo_RL_game-ezgif.com-video-to-gif-converter (2).gif"
        ),
        iio.imread(
            "/Users/antoinelaborde/Downloads/Demo_RL_game-ezgif.com-video-to-gif-converter (3).gif"
        ),
    ]
)
print("OK")

# get duration each frame is displayed
duration = iio.immeta(
    "/Users/antoinelaborde/Downloads/Demo_RL_game-ezgif.com-video-to-gif-converter.gif"
)["duration"]
print(duration)

iio.imwrite("combined.gif", frames, duration=duration)
